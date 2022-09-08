from copy import deepcopy
import torch
import torch.nn as nn

from models.anchors import AnchorGeneratorRotated
from models.init_weights import normal_init, bias_init_with_prob
from models.boxes import rboxes_encode, rboxes_decode

from utils.loss import SmoothL1Loss, FocalLoss

from utils.bbox_nms_rotated import multiclass_nms_rotated

from functools import partial

from models.backbone import DetectorBackbone
from models.neck import FPN, PAN

from models.alignconv import AlignConv

import math
from models.utils import assign_labels, split_to_levels, multi_apply

def multi_apply(func, *args, **kwargs):
    # 将函数func的参数kwargs固定，返回新的函数pfunc
    pfunc = partial(func, **kwargs) if kwargs else func
    # 这里的args表示feats和anchor_strides两个序列，map函数会分别遍历这两个序列，然后送入pfunc函数
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class RetinaNetHead(nn.Module):
    '''
    包括两部分：特征对齐模块(feature alignment module, FAM)、旋转检测模块(oriented detection module, ODM)
    input args:
        anchor_angles : 旋转anchor的角度设置，单位为弧度，由于S2ANet中角度的范围为[-0.25pi,0.75pi]，因此这个角度设置要格外注意
    '''
    def __init__(self, num_classes, in_channels=256, feat_channels=256, stacked_convs=2, 

        shared_head = True,
        
        # 使用置信度分支
        with_conf = True,

        # 是否使用对齐卷积进行分类
        with_alignconv = True,

        
        # 边界框的编解码方式，即xy坐标的编码，是否涉及到角度
        is_encode_relative = False,

        anchor_scales=[4],

        # anchor_ratios=[1.0],
        # anchor_angles = [0,],
        
        # anchor_ratios=[5.0, 8.0],
        anchor_ratios=[6.0],

        # anchor_angles = [0, 0.25*math.pi, 0.5*math.pi],
        # anchor_angles = [-0.25*math.pi, 0, 0.25*math.pi, 0.5*math.pi],
        anchor_angles = [-0.125*math.pi, 0.125*math.pi, 0.375*math.pi, 0.625*math.pi],


        featmap_strides=[8, 16, 32, 64, 128],

        score_thres_before_nms = 0.05,
        iou_thres_nms = 0.1,
        max_before_nms_per_level = 2000,
        max_per_img = 2000
    ):
        super().__init__()

        ## 输入图像的尺寸，主要用于计算损失时，对gt_boxes进行缩放，并且用于处理超出图像边界的anchors
        # (img_h, img_w)
        self.imgs_size = (1024, 1024)
        self.score_thres_before_nms = score_thres_before_nms # 进行NMS前阈值处理
        self.iou_thres_nms = iou_thres_nms                   # nms的iou阈值
        self.max_before_nms_per_level = max_before_nms_per_level     # 每个特征层级上进行NMS的检测框的个数
        self.max_per_img = max_per_img                       # 每张图像最多几个检测框

        self.num_classes = num_classes
        self.in_channels = in_channels      # 输入特征图的通道个数
        self.feat_channels = feat_channels  # head中间卷积层的输出通道数
        self.stacked_convs = stacked_convs  # head中间卷积层的个数，不包括最后输出结果的卷积层

        # 是否使用对齐卷积
        self.with_alignconv = with_alignconv   

        # 是否使用置信度预测分支
        self.with_conf = with_conf


        # 如果使用了对齐卷积，则必须使用置信度分支来挑选质量好的框
        if self.with_alignconv:
            assert self.with_conf

        # 是否使用共享head
        self.shared_head = shared_head
        # 旋转框xy坐标编解码是否涉及到角度
        self.is_encode_relative = is_encode_relative


        # FAM损失和ODM损失之间的平衡因子
        self.odm_balance = 1.0
        # FPN结构不同特征层级之间的平衡因子,这里先全部设置为1
        self.FPN_balance = (1.0, 1.0, 1.0, 1.0, 1.0) 
        # 分类损失和回归损失之间的平衡因子
        self.reg_balance = 1.0


        # anchor
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_angles = anchor_angles
        # 每个特征层级上的anchor的个数
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios) * len(self.anchor_angles)
        
        self.featmap_strides = featmap_strides
        
        # 特征层级个数
        self.num_levels = len(self.featmap_strides)

        # anchor的基础尺寸，即为特征层级的下采样倍数
        self.anchor_base_sizes = list(featmap_strides)
        
        self.anchor_generators = []
        for anchor_base_size in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGeneratorRotated(anchor_base_size, self.anchor_scales, self.anchor_ratios, angles=self.anchor_angles)
            )


        # S2ANet是基于RetinaNet的，不同特征层级共享head网络
        self._init_layers()
        self.init_weights()
        # 先对head初始化参数，然后每一个特征层级复制一份head，形成不共享的head网络
        self.create_unshared_head_layers()

        ## 损失函数是否创建的标志
        self.is_create_loss_func = False
        ## 损失函数定义，注意，论文中的损失，都是一个batch中所有样本的损失之和，然后除以正样本个数
        self.fl_gamma = 2.0
        self.fl_alpha = 0.5

        self.smoothL1_beta = 1.0 / 9.0

        

    def _init_loss_func(self):

        # 创建损失函数
        # 默认是'mean'状态
        loss_fam_cls = nn.BCEWithLogitsLoss(reduction='sum')
        loss_fam_cls = FocalLoss(loss_fam_cls, self.fl_gamma, self.fl_alpha) 

        self.loss_fam_cls = loss_fam_cls

        self.loss_fam_reg = SmoothL1Loss(beta=self.smoothL1_beta, reduction='sum')
        # self.loss_odm_reg = SmoothL1Loss(beta=self.smoothL1_beta, reduction='sum')

        if self.with_conf:
            loss_conf = nn.BCEWithLogitsLoss(reduction='sum')
            # loss_conf = FocalLoss(loss_conf, self.fl_gamma, self.fl_alpha) 
            self.loss_conf = loss_conf

        self.is_create_loss_func = True


    def _init_layers(self):
        # self.relu = nn.ReLU(inplace=True)
        # FAM模块和ODM模块的分类分支和回归分支
        reg_ls = []
        cls_ls = []

        for i in range(self.stacked_convs):
            in_chs = self.in_channels if i == 0 else self.feat_channels
            reg_ls.append(
                nn.Sequential(
                    nn.Conv2d(in_chs, self.feat_channels, 
                        kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True),
                    nn.ReLU(inplace=True)
                )
            )
            cls_ls.append(
                nn.Sequential(
                    nn.Conv2d(in_chs, self.feat_channels, 
                        kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True),
                    nn.ReLU(inplace=True)
                )
            )

        
        self.reg_ls = nn.Sequential(*reg_ls)
        self.cls_ls = nn.Sequential(*cls_ls)

        # FAM模块用于输出的卷积层，很奇怪，FAM用的是1x1的卷积，而ODM模块用的是3x3的卷积
        self.reg_head = nn.Conv2d(self.feat_channels, 5*self.num_anchors, kernel_size=(1,1), padding=0, bias=True)
        
        # 如果使用对齐卷积分类，则只需要一个anchor；否则，需要所有的anchors
        if not self.with_alignconv:
            self.cls_head = nn.Conv2d(self.feat_channels, self.num_classes*self.num_anchors, kernel_size=(1,1), padding=0, bias=True)
        else:
            self.cls_head = nn.Conv2d(self.feat_channels, self.num_classes, kernel_size=(1,1), padding=0, bias=True)


        # 对齐卷积
        if self.with_alignconv:
            self.align_conv = AlignConv(self.feat_channels, self.feat_channels, kernel_size=3)


        # 置信度分支与回归分支并行，只需要一个head就好了
        if self.with_conf:
            # FAM模块用于输出的卷积层，很奇怪，FAM用的是1x1的卷积，而ODM模块用的是3x3的卷积
            self.conf_head = nn.Conv2d(self.feat_channels, 1*self.num_anchors, kernel_size=(1,1), padding=0, bias=True)

    def create_unshared_head_layers(self):
        if not self.shared_head:
            self.reg_ls = nn.ModuleList(deepcopy(self.reg_ls) for i in range(self.num_levels))
            self.cls_ls = nn.ModuleList(deepcopy(self.cls_ls) for i in range(self.num_levels))

            self.reg_head = nn.ModuleList(deepcopy(self.reg_head) for i in range(self.num_levels))
            self.cls_head = nn.ModuleList(deepcopy(self.cls_head) for i in range(self.num_levels))

            if self.with_alignconv:
                self.align_conv = nn.ModuleList(deepcopy(self.align_conv) for i in range(self.num_levels))

            if self.with_conf:
                self.conf_head = nn.ModuleList(deepcopy(self.conf_head) for i in range(self.num_levels))


    def init_weights(self):
        
        bias_cls = bias_init_with_prob(0.01)

        for m in self.reg_ls.modules():
            t = type(m)
            if t is nn.Conv2d:
                normal_init(m, std=0.01)
        for m in self.cls_ls.modules():
            t = type(m)
            if t is nn.Conv2d:
                normal_init(m, std=0.01)
        normal_init(self.reg_head, std=0.01)
        normal_init(self.cls_head, std=0.01, bias=bias_cls)

        if self.with_alignconv:
            self.align_conv.init_weights()

        if self.with_conf:
            normal_init(self.conf_head, std=0.01, bias=bias_cls)


    def forward(self, feats, targets=None, imgs_size=None, post_process=False):
        # feats是个列表，存储每个层级的特征图; self.anchor_strides表示每个特征层级的下采样倍数
        # 返回的结果，是个元组，每个元组的元素是一个列表，具体形式如下：
        # ([从低特征层级到高层级的fam_cls_score, ...], [fam_bbox_pred,...], ...)
        if self.shared_head:
            p = multi_apply(self.forward_single, feats, self.featmap_strides)
        else:
            p = multi_apply(self.forward_single_unshared_head, feats, self.featmap_strides)
        
        # 即训练状态或者验证状态，总之需要计算损失函数
        if targets is not None:

            assert imgs_size is not None, "需要用imgs_size对targets处理获得像素为单位的标注框"

            self.imgs_size = imgs_size
            # 标注框转化为以像素为单位
            targets[:,[2,4]] = targets[:,[2,4]] * imgs_size[1]
            targets[:,[3,5]] = targets[:,[3,5]] * imgs_size[0]

            loss, loss_items = self.compute_loss(p, targets)

            # 不需要后处理步骤，即训练状态，只需要损失就好了，不需要进行边界框解码和NMS
            if not post_process:
                return loss, loss_items
            # 验证状态，即需要损失，也需要进行边界框解码和NMS去计算mAP指标
            else:
                imgs_results_ls = self.get_bboxes(p)
                return loss, loss_items, imgs_results_ls
        # 测试状态，或者推理状态，需要进行边界框解码和NMS去计算mAP指标
        else:
            if post_process:
                imgs_results_ls = self.get_bboxes(p)
                return imgs_results_ls
            else:
                return p
    
    # 经过一个特征层级的前向传播
    def forward_single(self, x, featmap_stride):
        # 查看是第几个特征层级，范围为P3-P7
        level_id = self.featmap_strides.index(featmap_stride)
        batch_size, _, feat_h, feat_w = x.shape
        # 高度和宽度，(H,W)
        featmap_size = (feat_h, feat_w)

        # 初始的anchor
        # init_anchors shape ： [H, W, num_anchors, 5(以像素为单位)]
        init_grid_anchors = self.anchor_generators[level_id].gen_grid_anchors(
            featmap_size, self.featmap_strides[level_id])
        init_grid_anchors = init_grid_anchors.to(x.device)



        # 网络的直接输出，没有经过激活函数
        reg_feat = self.reg_ls(x)
        fam_bbox_pred = self.reg_head(reg_feat)
        
        ## 调整fam_bbox_pred和fam_cls_pred的shape
        # [B, num_anchors*5, H, W], to [B, H, W, num_anchors, 5]
        fam_bbox_pred = fam_bbox_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)

        # 边界框解码
        bbox_pred_decode = self.get_bbox_decode(fam_bbox_pred.detach(), init_grid_anchors.clone())
            
        odm_cls_pred = None
        conf_pred = None
        refine_anchor = None
        index = None
        # 置信度分支
        if self.with_conf:
            conf_pred = self.conf_head(reg_feat)
            # [B, num_anchors*num_classes, H, W], to [B, H, W, num_anchors, 1]
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)

            # 根据初始的方形anchor，以及FAM的预测结果，得到修正后的旋转anchor
            # 这是一步边界框解码的过程，需要断开梯度的传递
            # 这里需要注意的是，fam_bbox_pred是没有经过激活函数的，就直接进行边界框解码了
            # 使用置信度挑选质量好的框
            # refine_anchor shape:[N, H, W, 5]
            refine_anchor, index, bbox_pred_decode = self.get_refine_anchors(conf_pred.detach(), bbox_pred_decode.clone())
        

        ## 分类分支
        if not self.with_alignconv:
            fam_cls_pred = self.cls_head(self.cls_ls(x))
            # [B, num_anchors*num_classes, H, W], to [B, H, W, num_anchors, num_classes]
            fam_cls_pred = fam_cls_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)
        else:
            fam_cls_pred = self.cls_head(self.cls_ls(
                # 根据FPN的特征图、修正后的旋转anchor，获得对齐后的特征图
                self.align_conv(x, refine_anchor.clone(), featmap_stride)
            ))
            # [B, num_anchors*num_classes, H, W], to [B, H, W, num_anchors=1, num_classes]
            fam_cls_pred = fam_cls_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, 1, -1)



        return fam_cls_pred, fam_bbox_pred, init_grid_anchors, odm_cls_pred, refine_anchor, conf_pred, bbox_pred_decode, index

    # 经过一个特征层级的前向传播
    def forward_single_unshared_head(self, x, featmap_stride):
        # 查看是第几个特征层级，范围为P3-P7
        level_id = self.featmap_strides.index(featmap_stride)
        batch_size, _, feat_h, feat_w = x.shape
        # 高度和宽度，(H,W)
        featmap_size = (feat_h, feat_w)

        # 初始的anchor
        # init_anchors shape ： [H, W, num_anchors, 5(以像素为单位)]
        init_grid_anchors = self.anchor_generators[level_id].gen_grid_anchors(
            featmap_size, self.featmap_strides[level_id])
        init_grid_anchors = init_grid_anchors.to(x.device)


        # 网络的直接输出，没有经过激活函数
        reg_feat = self.reg_ls[level_id](x)
        fam_bbox_pred = self.reg_head[level_id](reg_feat)
        
        ## 调整fam_bbox_pred和fam_cls_pred的shape
        # [B, num_anchors*5, H, W], to [B, H, W, num_anchors, 5]
        fam_bbox_pred = fam_bbox_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)


        # 边界框解码
        bbox_pred_decode = self.get_bbox_decode(fam_bbox_pred.detach(), init_grid_anchors.clone())

        odm_cls_pred = None
        conf_pred = None
        refine_anchor = None
        index = None
        # 置信度分支
        if self.with_conf:
            conf_pred = self.conf_head[level_id](reg_feat)
            # [B, num_anchors*num_classes, H, W], to [B, H, W, num_anchors, 1]
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)

            # 根据初始的方形anchor，以及FAM的预测结果，得到修正后的旋转anchor
            # 这是一步边界框解码的过程，需要断开梯度的传递
            # 这里需要注意的是，fam_bbox_pred是没有经过激活函数的，就直接进行边界框解码了
            # 使用置信度挑选质量好的框
            # refine_anchor shape:[N, H, W, 5]
            refine_anchor, index, bbox_pred_decode = self.get_refine_anchors(conf_pred.detach(), bbox_pred_decode.clone())
            

        ## 分类分支
        if not self.with_alignconv:
            fam_cls_pred = self.cls_head[level_id](self.cls_ls[level_id](x))
            # [B, num_anchors*num_classes, H, W], to [B, H, W, num_anchors, num_classes]
            fam_cls_pred = fam_cls_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)
        else:
            
            fam_cls_pred = self.cls_head[level_id](self.cls_ls[level_id](
                # 根据FPN的特征图、修正后的旋转anchor，获得对齐后的特征图
                self.align_conv[level_id](x, refine_anchor.clone(), featmap_stride)
            ))

            # [B, num_anchors*num_classes, H, W], to [B, H, W, num_anchors=1, num_classes]
            fam_cls_pred = fam_cls_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, 1, -1)


        return fam_cls_pred, fam_bbox_pred, init_grid_anchors, odm_cls_pred, refine_anchor, conf_pred, bbox_pred_decode, index


    def get_bbox_decode(self, fam_bbox_pred, init_grid_anchors):
        
        batch_size, feat_h, feat_w, _, _ = fam_bbox_pred.shape
        fam_bbox_pred = fam_bbox_pred.reshape(-1,5)

        num_anchors = self.num_anchors
        
        # init_grid_anchors shape: [H, W, num_anchors, 5] to [B, H, W, num_anchors, 5] 
        init_grid_anchors = init_grid_anchors[None, :].repeat(batch_size, 1, 1, 1, 1).reshape(-1,5)
        # 解码得到
        bbox_pred_decode = rboxes_decode(
                init_grid_anchors, fam_bbox_pred, is_encode_relative=self.is_encode_relative, wh_ratio_clip=1e-6)

        bbox_pred_decode = bbox_pred_decode.reshape(batch_size, feat_h, feat_w, num_anchors, -1)

        return bbox_pred_decode


    # 每个网格的位置出有多个anchors，选择置信度最高的anchors作为
    def get_refine_anchors(self, conf_pred, bbox_pred_decode):
        '''
            fam_cls_pred      : [B, H, W, num_anchors, num_classes]
            fam_bbox_pred     : [B, H, W, num_anchors, 5]
            init_grid_anchors : [H, W, num_anchors,5]
        '''
        num_anchors = self.num_anchors
        batch_size, feat_h, feat_w, _, _ = conf_pred.shape
        device = conf_pred.device

        # 如果FAM模块的每个特征点位置有多个anchors，那就根据分类分数，取这个位置点分数最大的anchor作为基准
        if num_anchors > 1:
            conf_pred = conf_pred.sigmoid()
            # 在类别通道找到最大值, fam_cls_pred_max_conf shape：[B,H,W,num_anchors]
            fam_cls_pred_max_conf, _ = conf_pred.max(dim=4)
            # 在num_anchors中找到置信度最大的anchors, max_conf_anchor_id shape :[B,H,W]
            _, max_conf_anchor_id = fam_cls_pred_max_conf.max(dim=3)
            max_conf_anchor_id = max_conf_anchor_id.reshape(-1)

            # 最大位置的索引，index:[B,H,W,num_anchors]
            # index = fam_cls_pred_max_conf == fam_cls_pred_max_conf_anchor[..., None]
            index = torch.zeros((batch_size*feat_h*feat_w, num_anchors), dtype=torch.bool, device=device)
            index[range(batch_size*feat_h*feat_w), max_conf_anchor_id] = True
            index = index.reshape(batch_size, feat_h, feat_w, num_anchors)

            # 在每个位置点的num_anchors个anchor中，只取出一个
            refine_anchor = bbox_pred_decode[index].reshape(batch_size, feat_h, feat_w, -1)
        
        # FAM模块每个位置点只有一个anchor，即S2ANet算法的设置
        elif num_anchors == 1:
            index = None
            refine_anchor = bbox_pred_decode.clone()
        else:
            raise NotImplementedError("num_anchors must be [1,inf)")

        return refine_anchor, index, bbox_pred_decode

    # 一张图像上的所有gt框，与所有特征层级的所有anchor，进行一次正负样本匹配
    def compute_loss(self, p, targets):
        '''
        p: ([从低特征层级到高层级的fam_cls_score, ...], [fam_bbox_pred,...], ...)
        targets: [num_object,7]，这个7的参数为[idx,c,x,y,w,h, theta]，idx为该目标所在的图像在batch中的索引号，xywh都是以像素为单位
        '''

        batch_size = p[0][0].shape[0]
        num_level = len(p[0])
        device = targets.device

        if not self.is_create_loss_func:
            self._init_loss_func()
        
        # 进行FAM模块和ODM模块的正负样本分配
        (fam_assign_gt_ids_levels, fam_pos_targets_levels, 
        fam_total_num_batchs_levels_pos, 
        odm_assign_gt_ids_levels, odm_pos_targets_levels, 
        odm_total_num_batchs_levels_pos, max_ious_levels,
        conf_total_num_batchs_levels_pos) = self.assign_labels_fam_odm(p, targets)


        cls_loss, reg_loss = torch.zeros(1, device=device), torch.zeros(1, device=device)
        odm_cls_loss, odm_reg_loss = torch.zeros(1, device=device), torch.zeros(1, device=device)
        # 置信度损失
        conf_loss = torch.zeros(1, device=device)
        ## 逐个特征层级，计算损失
        for level_id in range(num_level):
            
            # 计算FAM模块的损失
            # fam_cls_pred shape:[B, H, W, num_anchors, num_classes]
            fam_cls_pred = p[0][level_id]
            fam_cls_pred = fam_cls_pred.reshape(-1, self.num_classes)

            # fam_bbox_pred shape:[B, H, W, num_anchors, 5]
            fam_bbox_pred = p[1][level_id]
            fam_bbox_pred = fam_bbox_pred.reshape(-1, 5)
            
            # shape:[H, W, num_anchors, 5]
            init_grid_anchors = p[2][level_id]
            # shape:[B*H*W*num_anchors, 5]
            init_grid_anchors = init_grid_anchors.repeat(batch_size, 1, 1, 1, 1).reshape(-1,5)


            # 某个特征层级、整个batch的正负样本分配结果
            fam_assign_gt_ids_one_level = fam_assign_gt_ids_levels[level_id]
            fam_pos_targets_one_level = fam_pos_targets_levels[level_id]

            max_ious_one_level = None
            conf_pred = None
            if self.with_conf:
                conf_pred = p[5][level_id].reshape(-1)
                max_ious_one_level = max_ious_levels[level_id]

            index = None
            odm_assign_gt_ids_one_level = None
            odm_pos_targets_one_level = None
            if self.with_alignconv:
                index = p[7][level_id].reshape(-1)
                odm_assign_gt_ids_one_level = odm_assign_gt_ids_levels[level_id]
                odm_pos_targets_one_level = odm_pos_targets_levels[level_id]


            fam_cls_loss_single_level, fam_reg_loss_single_level, conf_loss_single_level = self.compute_loss_single_level(fam_bbox_pred, fam_cls_pred, init_grid_anchors, 
                fam_pos_targets_one_level, fam_assign_gt_ids_one_level, module_name="fam", max_ious_levels=max_ious_one_level,conf_pred=conf_pred, index=index, odm_assign_gt_ids=odm_assign_gt_ids_one_level, odm_pos_targets=odm_pos_targets_one_level)

            cls_loss += self.FPN_balance[level_id] * fam_cls_loss_single_level
            reg_loss += self.FPN_balance[level_id] * fam_reg_loss_single_level

            conf_loss += self.FPN_balance[level_id] * conf_loss_single_level


        # 除以整个batch、所有特征层级的正样本的个数
        if not self.with_alignconv:
            cls_loss /= fam_total_num_batchs_levels_pos
        else:
            cls_loss /= odm_total_num_batchs_levels_pos
        
        reg_loss /= fam_total_num_batchs_levels_pos
        
        # 回归损失的加权，以平衡分类损失
        reg_loss *= self.reg_balance

        total_loss = cls_loss + reg_loss
        
        
        if self.with_conf:
            conf_loss /= conf_total_num_batchs_levels_pos
            total_loss = total_loss + conf_loss
            
        return total_loss, torch.cat((cls_loss,reg_loss,odm_cls_loss,odm_reg_loss, conf_loss)).detach().cpu().numpy()


    # 进行FAM模块和ODM模块的正负样本分配
    def assign_labels_fam_odm(self, p, targets):
        
        batch_size = p[0][0].shape[0]
        num_level = len(p[0])


        # 先进行正负样本分配，逐张图像进行
        # 一个图像上的所有gt框，与一个图像上所有特征层级的所有网格anchors，进行正样本分配，确保一个目标不会分配到多个特征层级上
        init_grid_anchors_every_level = p[2]
        # init_grid_anchors shape:[H,W,num_anchors,5]
        init_grid_anchors_every_level = [init_grid_anchors.reshape(-1,5) for init_grid_anchors in init_grid_anchors_every_level]
        
        # 保存每一个特征层级的网格anchors的个数
        num_grid_anchors_every_level = [init_grid_anchors.shape[0] for init_grid_anchors in init_grid_anchors_every_level]
        # 一张图像的所有层级的网格anchor拼接为一个tensor
        init_grid_anchors_all_levels = torch.cat(init_grid_anchors_every_level, dim=0)

        # 存储正负样本分配的结果，二级列表，第一级存储各个特征层级，第二级存储各个batch
        fam_assign_gt_ids_levels_batch = [ [] for _ in range(num_level)]
        fam_pos_targets_levels_batch = [ [] for _ in range(num_level)]


        if self.with_conf:
            # 每一个列表元素的shape为[N, feat_h,feat_w, num_anchors, 5]
            bbox_pred_decode_every_level = p[6]
            bbox_pred_decode_batch_levels = [ [] for _ in range(batch_size)]  
            # 每一个特征层级的anchors的个数
            for batch_id in range(batch_size):
                for level_id in range(num_level):
                    # 获得一个特征层级、一个图像的精炼anchors
                    bbox_pred_decode = bbox_pred_decode_every_level[level_id][batch_id].reshape(-1,5)

                    bbox_pred_decode_batch_levels[batch_id].append(bbox_pred_decode)
            bbox_pred_decode_batch_levels = [torch.cat(i, dim=0) for i in bbox_pred_decode_batch_levels]

            conf_assign_gt_ids_levels_batch = [ [] for _ in range(num_level)]

            # 存储IOU
            max_ious_levels_batch = [ [] for _ in range(num_level)]


        if self.with_alignconv:
            # 每一个列表元素的shape为[N,feat_h,feat_w,5]
            refine_anchors_levels = p[4]
            refine_anchors_batch_levels = [ [] for _ in range(batch_size)]  
            # 每一个特征层级的anchors的个数
            num_refine_anchors_every_level = [(refine_anchors_one_level.shape[1:3]).numel() for refine_anchors_one_level in refine_anchors_levels]        
            for batch_id in range(batch_size):
                for level_id in range(num_level):
                    # 获得一个特征层级、一个图像的精炼anchors
                    refine_anchors = refine_anchors_levels[level_id][batch_id].reshape(-1,5)
                    
                    refine_anchors_batch_levels[batch_id].append(refine_anchors)

            refine_anchors_batch_levels = [torch.cat(i, dim=0) for i in refine_anchors_batch_levels]
            odm_assign_gt_ids_levels_batch = [ [] for _ in range(num_level)]
            odm_pos_targets_levels_batch = [ [] for _ in range(num_level)]

        # 逐张图像进行正负样本分配
        for batch_id in range(batch_size):
            # 取出该图像中的标注框
            # 注意，targets_one_img 的shape可能为(0,7)，标签分配时需要考虑到这一点
            targets_one_img = targets[targets[:,0]==batch_id].reshape(-1,7)

            '''进行fam模块的正负样本分配'''
            # 一张图像上的所有gt框，与所有特征层级、所有网格anchors进行正负样本分配，确保一个目标不会分配到多个特征层级上
            fam_assign_gt_ids, _ = assign_labels(init_grid_anchors_all_levels, targets_one_img[:, 2:7], imgs_size=self.imgs_size)

            # 将正负样本分配结果，拆分为各个特征层级的分配结果
            fam_assign_gt_ids_levels = split_to_levels(fam_assign_gt_ids, num_grid_anchors_every_level)

            for level_id in range(num_level):
                # 该特征层级的正负样本分配结果
                fam_assign_gt_ids_one_level = fam_assign_gt_ids_levels[level_id]
                # 保存该图像、该特征层级的正负样本分配结果
                fam_assign_gt_ids_levels_batch[level_id].append(fam_assign_gt_ids_one_level)
        
                # 将对应的gt框，也按所在的特征层级存储起来
                # 取出正样本位置
                fam_pos_gt_ids_one_level = fam_assign_gt_ids_one_level >= 0
                # 取出正样本对应的gt框
                fam_pos_targets_one_level = targets_one_img[fam_assign_gt_ids_one_level[fam_pos_gt_ids_one_level]].reshape(-1,7)
                fam_pos_targets_levels_batch[level_id].append(fam_pos_targets_one_level)
            

            # 置信度分支，需要使用所有的解码后的bbox_pred与gt进行正负样本分配
            if self.with_conf:
                bbox_pred_decode_one_img = bbox_pred_decode_batch_levels[batch_id]
                conf_assign_gt_ids, max_ious = assign_labels(bbox_pred_decode_one_img, targets_one_img[:, 2:7], imgs_size=self.imgs_size)
                
                conf_assign_gt_ids_levels = split_to_levels(conf_assign_gt_ids, num_grid_anchors_every_level)

                # 拆分为各个特征层级上的、每个anchor的最大iou
                max_ious_levels = split_to_levels(max_ious, num_grid_anchors_every_level)
                for level_id in range(num_level):
                    # 该特征层级的正负样本分配结果
                    conf_assign_gt_ids_one_level = conf_assign_gt_ids_levels[level_id]
                    # 保存该图像、该特征层级的正负样本分配结果
                    conf_assign_gt_ids_levels_batch[level_id].append(conf_assign_gt_ids_one_level)

                    max_ious_levels_batch[level_id].append(max_ious_levels[level_id])
            
            # 如果使用对齐卷积，需要使用refine_anchors与gt进行正负样本分配
            if self.with_alignconv:
                '''然后进行odm模块的正负样本分配'''
                # 取出该张图像，所有特征层级的精调anchors
                refine_anchor_one_img = refine_anchors_batch_levels[batch_id]
                # 该张图像上的所有gt框，和所有特征层级的精调anchors进行正负样本分配
                odm_assign_gt_ids, _ = assign_labels(refine_anchor_one_img, targets_one_img[:, 2:7], imgs_size=self.imgs_size)

                # 将正负样本分配结果，拆分为各个特征层级的分配结果
                odm_assign_gt_ids_levels = split_to_levels(odm_assign_gt_ids, num_refine_anchors_every_level)

                for level_id in range(num_level):
                    # 该特征层级的正负样本分配结果
                    odm_assign_gt_ids_one_level = odm_assign_gt_ids_levels[level_id]
                    # 保存该图像、该特征层级的正负样本分配结果
                    odm_assign_gt_ids_levels_batch[level_id].append(odm_assign_gt_ids_one_level)
            
                    # 将对应的gt框，也按所在的特征层级存储起来
                    # 取出正样本位置
                    odm_pos_gt_ids_one_level = odm_assign_gt_ids_one_level >= 0
                    # 取出正样本对应的gt框
                    odm_pos_targets_one_level = targets_one_img[odm_assign_gt_ids_one_level[odm_pos_gt_ids_one_level]].reshape(-1,7)
                    odm_pos_targets_levels_batch[level_id].append(odm_pos_targets_one_level)


        # 逐个特征层级的分配结果进行合并
        # fam_assign_gt_ids_levels shape : (batch_size*num_anchors_one_level)
        fam_assign_gt_ids_levels = [torch.cat(i, dim=0) for i in fam_assign_gt_ids_levels_batch]
        # 整个batch中的图像、所有特征层级的正样本的个数
        fam_total_num_batchs_levels_pos = self.get_total_num_pos_batch_levels(fam_assign_gt_ids_levels, batch_size)
        fam_pos_targets_levels = []
        for i in fam_pos_targets_levels_batch:
            if len(i):
                fam_pos_targets_levels.append(torch.cat(i, dim=0))
            else:
                fam_pos_targets_levels.append(None)


        max_ious_levels = None
        conf_total_num_batchs_levels_pos = None
        if self.with_conf:
            max_ious_levels = [torch.cat(i, dim=0) for i in max_ious_levels_batch]

            conf_assign_gt_ids_levels = [torch.cat(i, dim=0) for i in conf_assign_gt_ids_levels_batch]
            # 整个batch的图像中所有特征层级、所有正样本的数量
            conf_total_num_batchs_levels_pos = self.get_total_num_pos_batch_levels(conf_assign_gt_ids_levels, batch_size)
            # conf_total_num_batchs_levels_pos = sum((i>=0.35).sum().item() for i in max_ious_levels)
            # conf_total_num_batchs_levels_pos = max(conf_total_num_batchs_levels_pos, batch_size)
            
        
        odm_assign_gt_ids_levels, odm_pos_targets_levels, odm_total_num_batchs_levels_pos = None, None, None
        if self.with_alignconv:
            odm_assign_gt_ids_levels = [torch.cat(i, dim=0) for i in odm_assign_gt_ids_levels_batch]
            # 整个batch的图像中所有特征层级、所有正样本的数量
            odm_total_num_batchs_levels_pos = self.get_total_num_pos_batch_levels(odm_assign_gt_ids_levels, batch_size)
            odm_pos_targets_levels = []
            for i in odm_pos_targets_levels_batch:
                if len(i):
                    odm_pos_targets_levels.append(torch.cat(i, dim=0))
                else:
                    odm_pos_targets_levels.append(None)

        return fam_assign_gt_ids_levels, fam_pos_targets_levels, fam_total_num_batchs_levels_pos, \
               odm_assign_gt_ids_levels, odm_pos_targets_levels, odm_total_num_batchs_levels_pos, max_ious_levels, conf_total_num_batchs_levels_pos   


    # 获得整个batch、所有特征层级、所有正样本的总数
    def get_total_num_pos_batch_levels(self, assign_gt_ids_levels, batch_size):
        total_num = sum((i>=0).sum().item() for i in assign_gt_ids_levels)
        # 总数最小为batch_size
        total_num = max(total_num, batch_size)

        return total_num


    # 计算一个特征层级上的损失，包括回归损失和分类损失
    def compute_loss_single_level(self, bbox_pred, cls_pred, anchors, pos_targets_batch, assign_gt_ids_batch, module_name,conf_pred=None, max_ious_levels=None, index=None, odm_assign_gt_ids=None, odm_pos_targets=None):
        '''
            bbox_pred : [B*H*W*num_anchors, 5(x,y,w,h,theta)]，回归网络的直接输出,没有经过激活函数、边界框编解码
            cls_pred  : [B*H*W*num_anchors, num_classes] ，分类网络的直接输出,没有经过激活函数
            anchors   : [B*H*W*num_anchors, 5]，所有图像、所有网格的anchors
            pos_targets_batch : [N(正样本个数), 7(batch_id, cls_id, x,y,w,h,theta)]，x,y,w,h单位是像素值，theta单位是弧度
            assign_gt_ids_batch : (B*H*W*num_anchors), >=0表示正样本的类别id，-1表示负样本，-2表示忽略样本

        '''
        assert module_name in ("fam", "odm")

        device = anchors.device
        cls_loss, reg_loss = torch.zeros(1, device=device), torch.zeros(1, device=device)
        # 正样本的位置
        pos_gt_ids_batch = assign_gt_ids_batch >= 0
        # 正样本个数
        num_pos = pos_gt_ids_batch.sum().item()

        # 如果存在目标，并且有正样本
        if pos_targets_batch is not None and num_pos>0:
            
            if module_name == 'fam':
                #### 计算回归损失
                # 取出正样本对应的anchor
                pos_anchors_batch = anchors[pos_gt_ids_batch]
                ## 计算回归损失，回归损失只计算正样本
                # 计算gt_boxes对应的回归标签，即对gt进行编码
                pos_reg_targets_batch = rboxes_encode(pos_anchors_batch, pos_targets_batch[:,2:7], is_encode_relative=self.is_encode_relative)

                # 把正样本对应位置的预测取出来
                pos_bbox_pred_batch = bbox_pred[pos_gt_ids_batch]
                # 计算回归损失
                # pos_fam_bbox_pred_batch: torch.float16, fam_pos_reg_targets_batch:torch.float32
                reg_loss = self.loss_fam_reg(pos_bbox_pred_batch, pos_reg_targets_batch) 

            if not self.with_alignconv:
                # 计算正样本的分类损失
                # 正样本对应的分类预测值
                pos_cls_pred_batch = cls_pred[pos_gt_ids_batch]
                # 正样本对应的分类标签值
                pos_cls_targets_batch = torch.zeros_like(pos_cls_pred_batch)
                # 正样本类别标签所在通道设置为1
                pos_cls_targets_batch[range(num_pos), pos_targets_batch[:,1].long()] = 1

                cls_loss = self.loss_fam_cls(pos_cls_pred_batch, pos_cls_targets_batch)

        else:
            # 说明整个batch中都没有一个gt框，也没有正样本
            # 这种情况下，回归网络分支的参数和预测结果都不会参与损失的计算，在DDP模式下可能会报错
            pass
        
        if not self.with_alignconv:
            # 负样本的分类损失
            neg_gt_ids_batch = assign_gt_ids_batch == -1
            num_neg = neg_gt_ids_batch.sum().item()
            if num_neg>0:
                # 负样本对应的分类预测值
                neg_cls_pred_batch = cls_pred[neg_gt_ids_batch]
                # 负样本对应的分类标签值
                neg_cls_targets_batch = torch.zeros_like(neg_cls_pred_batch)

                cls_loss += self.loss_fam_cls(neg_cls_pred_batch, neg_cls_targets_batch)


        conf_loss = torch.zeros(1,device=device)
        if self.with_conf:
            conf_targets = max_ious_levels.detach()
            mask = (conf_targets >0) & (conf_targets < 1)
            if mask.sum()>0:
                # print(f"conf_pred:{conf_pred.shape}")
                # print(f"conf_targets:{conf_targets.shape}")
                conf_loss = self.loss_conf(conf_pred[mask], conf_targets[mask])
        

        # # 对齐分类损失，需要挑选出合适的anchors进行损失的计算，每个特征点的多个anchors中只取出一个anchors
        if self.with_alignconv:

            pos_gt_ids_batch = odm_assign_gt_ids >= 0
            # 把被选中的anchors中对应的正样本标签提取出来
            
            # 正样本个数
            num_pos = pos_gt_ids_batch.sum().item()

            # 如果存在目标，并且有正样本
            if odm_pos_targets is not None and num_pos>0:

                # 正样本对应的分类预测值
                pos_cls_pred_batch = cls_pred[pos_gt_ids_batch]
                # 正样本对应的分类标签值
                pos_cls_targets_batch = torch.zeros_like(pos_cls_pred_batch)
                # 正样本类别标签所在通道设置为1
                pos_cls_targets_batch[range(num_pos), odm_pos_targets[:,1].long()] = 1

                cls_loss = self.loss_fam_cls(pos_cls_pred_batch, pos_cls_targets_batch)
            

            # 负样本的分类损失
            neg_gt_ids_batch = odm_assign_gt_ids == -1
            num_neg = neg_gt_ids_batch.sum().item()
            if num_neg>0:
                # 负样本对应的分类预测值
                neg_cls_pred_batch = cls_pred[neg_gt_ids_batch]
                # 负样本对应的分类标签值
                neg_cls_targets_batch = torch.zeros_like(neg_cls_pred_batch)

                cls_loss += self.loss_fam_cls(neg_cls_pred_batch, neg_cls_targets_batch)


        return cls_loss, reg_loss, conf_loss

 

    # 对网络预测结果进行后处理，包括边界框解码、NMS，获得输入图像尺寸上的边界框坐标
    def get_bboxes(self, p, module_name="fam"):
        '''
        module_name : 是使用ODM模块，还是使用FAM模块的预测结果，作为最终的检测结果

        '''
        assert module_name in ('fam', 'odm'), "must be FAM or ODM"

        batch_size = p[0][0].shape[0]
        num_level = len(p[0])

        # FAM模块
        # cls_pred shape : [B, H, W, num_anchors, num_classes]
        cls_pred = p[0]
        # bbox_pred_decode  shape : [B, H, W, num_anchors, 5]
        bbox_pred_decode = p[6]        

        # 使用置信度，作为边界框NMS的分数
        if self.with_conf:
            # 使用置信度分支
            conf_pred = p[5]

        if self.with_alignconv:

            assert self.num_anchors > 1
            # 筛选出单个anchors，以及对应的置信度
            index_all = p[7]
            # p[4]为refine_anchors
            bbox_pred_decode = p[4]
            conf_choosed_pred = []
            for conf_pred_single_level, index_single_level in zip(conf_pred, index_all):
                
                _, feat_h, feat_w, _, _ = conf_pred_single_level.shape

                conf_choosed_pred.append(conf_pred_single_level[index_single_level].reshape(batch_size, feat_h, feat_w, -1))
                conf_pred = conf_choosed_pred

        # 检测框的结果
        imgs_results_ls = []
        for batch_id in range(batch_size):
            
            # 获得该张图像上的各个特征层级的预测结果
            scores_levels = []
            bbox_pred_decode_levels = []
            for level_id in range(num_level):
                if self.with_conf:
                    score_one_img_one_level = cls_pred[level_id][batch_id].detach().reshape(-1, self.num_classes).sigmoid() * \
                        conf_pred[level_id][batch_id].detach().reshape(-1, 1).sigmoid()
                else:
                    score_one_img_one_level = cls_pred[level_id][batch_id].detach().reshape(-1, self.num_classes).sigmoid()
                bbox_pred_deoce_one_img_one_level = bbox_pred_decode[level_id][batch_id].detach().reshape(-1, 5)

                scores_levels.append(score_one_img_one_level)
                bbox_pred_decode_levels.append(bbox_pred_deoce_one_img_one_level)

            # 进行一张图像的NMS
            det_bboxes, det_labels = self.get_bboxes_single_img(scores_levels, bbox_pred_decode_levels)

            imgs_results_ls.append((det_bboxes, det_labels))
        
        return imgs_results_ls

    def get_bboxes_single_img(self, scores_levels, bbox_pred_decode_levels):
        
        # 在进行NMS之前，要先过滤掉过多的检测框
        # 注意！是对每个特征层级上的预测框个数进行限制，而不是对整个图像上的预测框进行限制
        max_before_nms_single_level = self.max_before_nms_per_level

        # 存储一张图像上所有特征层级的、经过过滤后的预测分数、框和anchor
        scores = []
        bboxes = []
        # 逐个特征层级进行处理
        for score_level, bbox_pred_decode_level in zip(scores_levels, bbox_pred_decode_levels):
            # 在NMS前，根据分类分数进行阈值处理，过滤掉过多的框
            if max_before_nms_single_level > 0 and score_level.shape[0] > max_before_nms_single_level:
                max_scores, _ = score_level.max(dim=1)
                _, topk_inds = max_scores.topk(max_before_nms_single_level)

                score_level = score_level[topk_inds, :] # shape:[N, num_classes]
                bbox_pred_decode_level = bbox_pred_decode_level[topk_inds, :]     # shape:[N, 5]
            
            scores.append(score_level)
            bboxes.append(bbox_pred_decode_level)
        
        ## 不同层级的预测结果拼接成一个tensor
        scores = torch.cat(scores, dim=0)
        bboxes = torch.cat(bboxes, dim=0)

        det_bboxes, det_labels = multiclass_nms_rotated(bboxes, scores, 
                score_thr = self.score_thres_before_nms, 
                iou_thr = self.iou_thres_nms, 
                max_per_img = self.max_per_img
            )

        return det_bboxes, det_labels
        
   
class RetinaNet(nn.Module):
    def __init__(self, backbone_name="resnet50", num_classes=15):
        super().__init__()

        # 用于检测的每个特征层级的下采样次数
        # self.stride = [4, 8, 16, 32]
        self.stride = [8, 16, 32]
        # self.stride = [8, 16, 32, 64 ,128]
        self.nl = len(self.stride)  # 检测层的个数，即neck网络输出的特征层级的个数

        # self.backbone_out_out_indices = (1,2,3,4)
        # backbone输出C3、C4、C5三个特征图
        self.backbone = DetectorBackbone(backbone_name)

        # self.neck = FPN(
        #     in_channels=[512,1024,2048],
        #     num_outs=self.nl
        # )
        self.neck = PAN(
            in_channels=[512,1024,2048],
            num_outs=self.nl
        )

        # self.neck = BiFPN(
        #     in_channels=[512,1024,2048],
        #     # num_outs=self.nl
        # )
        
        self.head = RetinaNetHead(num_classes=num_classes, featmap_strides=self.stride)
        


    def forward(self, imgs, targets=None, post_process=False):

        outs = self.backbone(imgs)
        outs = self.neck(outs)
        
        if targets is not None:
            
            # 不需要后处理，训练状态
            if not post_process:
                loss, loss_items = self.head(outs, targets=targets, imgs_size=imgs.shape[-2:], post_process=post_process)
                return loss, loss_items
            else:
                loss, loss_items, imgs_results_ls = self.head(outs, targets=targets, imgs_size=imgs.shape[-2:], post_process=post_process)
                return loss, loss_items, imgs_results_ls
        # 测试状态，不需要损失，但是需要进行后处理计算mAP
        else:
            
            if post_process:
                imgs_results_ls = self.head(outs, post_process=post_process)
                return imgs_results_ls
            else:
                p = self.head(outs, post_process=post_process)
                return p

