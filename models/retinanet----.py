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
    
    pfunc = partial(func, **kwargs) if kwargs else func
    
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class RetinaNetHead(nn.Module):
    def __init__(self, num_classes, in_channels=256, feat_channels=256, stacked_convs=2, 

        shared_head = True,
        with_alignconv = True,
        with_conf = True,
        is_encode_relative = False,
        anchor_scales=[4],
        anchor_ratios=[6.0],
        anchor_angles = [-0.125*math.pi, 0.125*math.pi, 0.375*math.pi, 0.625*math.pi],
        featmap_strides=[8, 16, 32, 64, 128],
        score_thres_before_nms = 0.05,
        iou_thres_nms = 0.1,
        max_before_nms_per_level = 2000,
        max_per_img = 2000
    ):
        super().__init__()

        self.imgs_size = (1024, 1024)
        self.score_thres_before_nms = score_thres_before_nms 
        self.iou_thres_nms = iou_thres_nms                   
        self.max_before_nms_per_level = max_before_nms_per_level     
        self.max_per_img = max_per_img                       

        self.num_classes = num_classes
        self.in_channels = in_channels      
        self.feat_channels = feat_channels  
        self.stacked_convs = stacked_convs  
        self.with_alignconv = with_alignconv      
        self.with_conf = with_conf

        if self.with_alignconv:
            assert self.with_conf

        self.shared_head = shared_head
        self.is_encode_relative = is_encode_relative
        self.odm_balance = 1.0
        self.FPN_balance = (1.0, 1.0, 1.0, 1.0, 1.0) 
        
        self.reg_balance = 1.0

        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_angles = anchor_angles
        
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios) * len(self.anchor_angles)
        
        self.featmap_strides = featmap_strides

        self.num_levels = len(self.featmap_strides)

        self.anchor_base_sizes = list(featmap_strides)
        
        self.anchor_generators = []
        for anchor_base_size in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGeneratorRotated(anchor_base_size, self.anchor_scales, self.anchor_ratios, angles=self.anchor_angles)
            )

        self._init_layers()
        self.init_weights()
        
        self.create_unshared_head_layers()

        self.is_create_loss_func = False
        
        self.fl_gamma = 2.0
        self.fl_alpha = 0.5

        self.smoothL1_beta = 1.0 / 9.0
    def _init_loss_func(self):

        loss_fam_cls = nn.BCEWithLogitsLoss(reduction='sum')
        loss_fam_cls = FocalLoss(loss_fam_cls, self.fl_gamma, self.fl_alpha) 

        self.loss_fam_cls = loss_fam_cls

        self.loss_fam_reg = SmoothL1Loss(beta=self.smoothL1_beta, reduction='sum')
        
        if self.with_conf:
            loss_conf = nn.BCEWithLogitsLoss(reduction='sum')
            
            self.loss_conf = loss_conf

        self.is_create_loss_func = True
    def _init_layers(self):

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

        self.reg_head = nn.Conv2d(self.feat_channels, 5*self.num_anchors, kernel_size=(1,1), padding=0, bias=True)

        if not self.with_alignconv:
            self.cls_head = nn.Conv2d(self.feat_channels, self.num_classes*self.num_anchors, kernel_size=(1,1), padding=0, bias=True)
        else:
            self.cls_head = nn.Conv2d(self.feat_channels, self.num_classes, kernel_size=(1,1), padding=0, bias=True)

        if self.with_alignconv:
            self.align_conv = AlignConv(
                self.feat_channels, self.feat_channels, kernel_size=3)

        if self.with_conf:
            
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
        if self.shared_head:
            p = multi_apply(self.forward_single, feats, self.featmap_strides)
        else:
            p = multi_apply(self.forward_single_unshared_head, feats, self.featmap_strides)

        if targets is not None:

            assert imgs_size is not None, "需要用imgs_size对targets处理获得像素为单位的标注框"

            self.imgs_size = imgs_size
            
            targets[:,[2,4]] = targets[:,[2,4]] * imgs_size[1]
            targets[:,[3,5]] = targets[:,[3,5]] * imgs_size[0]

            loss, loss_items = self.compute_loss(p, targets)

            if not post_process:
                return loss, loss_items 
            else:
                imgs_results_ls = self.get_bboxes(p)
                return loss, loss_items, imgs_results_ls
        else:
            if post_process:
                imgs_results_ls = self.get_bboxes(p)
                return imgs_results_ls
            else:
                return p
    
    
    def forward_single(self, x, featmap_stride):
        
        level_id = self.featmap_strides.index(featmap_stride)
        batch_size, _, feat_h, feat_w = x.shape
        
        featmap_size = (feat_h, feat_w)
        init_grid_anchors = self.anchor_generators[level_id].gen_grid_anchors(
            featmap_size, self.featmap_strides[level_id])
        init_grid_anchors = init_grid_anchors.to(x.device)
        reg_feat = self.reg_ls(x)
        fam_bbox_pred = self.reg_head(reg_feat)

        fam_bbox_pred = fam_bbox_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)

        bbox_pred_decode = self.get_bbox_decode(fam_bbox_pred.detach(), init_grid_anchors.clone())
            
        odm_cls_pred = None
        conf_pred = None
        refine_anchor = None
        index = None
        
        if self.with_conf:
            conf_pred = self.conf_head(reg_feat)
            
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)

            refine_anchor, index, bbox_pred_decode = self.get_refine_anchors(conf_pred.detach(), bbox_pred_decode.clone())

        if not self.with_alignconv:
            fam_cls_pred = self.cls_head(self.cls_ls(x))
            
            fam_cls_pred = fam_cls_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)
        else:
            
            fam_cls_pred = self.cls_head(self.cls_ls(
                
                self.align_conv(x, refine_anchor.clone(), featmap_stride)
            ))
 
            fam_cls_pred = fam_cls_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, 1, -1)
        return fam_cls_pred, fam_bbox_pred, init_grid_anchors, odm_cls_pred, refine_anchor, conf_pred, bbox_pred_decode, index

    
    def forward_single_unshared_head(self, x, featmap_stride):
        
        level_id = self.featmap_strides.index(featmap_stride)
        batch_size, _, feat_h, feat_w = x.shape
        
        featmap_size = (feat_h, feat_w)
        init_grid_anchors = self.anchor_generators[level_id].gen_grid_anchors(
            featmap_size, self.featmap_strides[level_id])
        init_grid_anchors = init_grid_anchors.to(x.device)

        reg_feat = self.reg_ls[level_id](x)
        fam_bbox_pred = self.reg_head[level_id](reg_feat)

        fam_bbox_pred = fam_bbox_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)
        bbox_pred_decode = self.get_bbox_decode(fam_bbox_pred.detach(), init_grid_anchors.clone())

        odm_cls_pred = None
        conf_pred = None
        refine_anchor = None
        index = None
        
        if self.with_conf:
            conf_pred = self.conf_head[level_id](reg_feat)
            
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)
            refine_anchor, index, bbox_pred_decode = self.get_refine_anchors(conf_pred.detach(), bbox_pred_decode.clone())
        
        if not self.with_alignconv:
            fam_cls_pred = self.cls_head[level_id](self.cls_ls[level_id](x))
            
            fam_cls_pred = fam_cls_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)
        else:
            
            fam_cls_pred = self.cls_head[level_id](self.cls_ls[level_id](
                
                self.align_conv[level_id](x, refine_anchor.clone(), featmap_stride)
            ))
            fam_cls_pred = fam_cls_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, 1, -1)

        return fam_cls_pred, fam_bbox_pred, init_grid_anchors, odm_cls_pred, refine_anchor, conf_pred, bbox_pred_decode, index


    def get_bbox_decode(self, fam_bbox_pred, init_grid_anchors):
        
        batch_size, feat_h, feat_w, _, _ = fam_bbox_pred.shape
        fam_bbox_pred = fam_bbox_pred.reshape(-1,5)

        num_anchors = self.num_anchors
        init_grid_anchors = init_grid_anchors[None, :].repeat(batch_size, 1, 1, 1, 1).reshape(-1,5)
        
        bbox_pred_decode = rboxes_decode(
                init_grid_anchors, fam_bbox_pred, is_encode_relative=self.is_encode_relative, wh_ratio_clip=1e-6)

        bbox_pred_decode = bbox_pred_decode.reshape(batch_size, feat_h, feat_w, num_anchors, -1)

        return bbox_pred_decode

    def get_refine_anchors(self, conf_pred, bbox_pred_decode):

        num_anchors = self.num_anchors
        batch_size, feat_h, feat_w, _, _ = conf_pred.shape
        device = conf_pred.device
        if num_anchors > 1:
            conf_pred = conf_pred.sigmoid()
            
            fam_cls_pred_max_conf, _ = conf_pred.max(dim=4)
            
            _, max_conf_anchor_id = fam_cls_pred_max_conf.max(dim=3)
            max_conf_anchor_id = max_conf_anchor_id.reshape(-1)

            index = torch.zeros((batch_size*feat_h*feat_w, num_anchors), dtype=torch.bool, device=device)
            index[range(batch_size*feat_h*feat_w), max_conf_anchor_id] = True
            index = index.reshape(batch_size, feat_h, feat_w, num_anchors)

            refine_anchor = bbox_pred_decode[index].reshape(batch_size, feat_h, feat_w, -1)
        elif num_anchors == 1:
            index = None
            refine_anchor = bbox_pred_decode.clone()
        else:
            raise NotImplementedError("num_anchors must be [1,inf)")
        return refine_anchor, index, bbox_pred_decode

    
    def compute_loss(self, p, targets):
        batch_size = p[0][0].shape[0]
        num_level = len(p[0])
        device = targets.device

        if not self.is_create_loss_func:
            self._init_loss_func()
        
        (fam_assign_gt_ids_levels, fam_pos_targets_levels, 
        fam_total_num_batchs_levels_pos, 
        odm_assign_gt_ids_levels, odm_pos_targets_levels, 
        odm_total_num_batchs_levels_pos, max_ious_levels,
        conf_total_num_batchs_levels_pos) = self.assign_labels_fam_odm(p, targets)


        cls_loss, reg_loss = torch.zeros(1, device=device), torch.zeros(1, device=device)
        odm_cls_loss, odm_reg_loss = torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        conf_loss = torch.zeros(1, device=device)
        
        for level_id in range(num_level):
            fam_cls_pred = p[0][level_id]
            fam_cls_pred = fam_cls_pred.reshape(-1, self.num_classes)

            fam_bbox_pred = p[1][level_id]
            fam_bbox_pred = fam_bbox_pred.reshape(-1, 5)

            init_grid_anchors = p[2][level_id]
            
            init_grid_anchors = init_grid_anchors.repeat(batch_size, 1, 1, 1, 1).reshape(-1,5)
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
        
        if not self.with_alignconv:
            cls_loss /= fam_total_num_batchs_levels_pos
        else:
            cls_loss /= odm_total_num_batchs_levels_pos
        
        reg_loss /= fam_total_num_batchs_levels_pos 
        reg_loss *= self.reg_balance
        total_loss = cls_loss + reg_loss
        
        if self.with_conf:
            conf_loss /= conf_total_num_batchs_levels_pos
            total_loss = total_loss + conf_loss
            
        return total_loss, torch.cat((cls_loss,reg_loss,odm_cls_loss,odm_reg_loss, conf_loss)).detach().cpu().numpy()

    def assign_labels_fam_odm(self, p, targets):
        
        batch_size = p[0][0].shape[0]
        num_level = len(p[0])

        init_grid_anchors_every_level = p[2]
        
        init_grid_anchors_every_level = [init_grid_anchors.reshape(-1,5) for init_grid_anchors in init_grid_anchors_every_level]
        num_grid_anchors_every_level = [init_grid_anchors.shape[0] for init_grid_anchors in init_grid_anchors_every_level]
        
        init_grid_anchors_all_levels = torch.cat(init_grid_anchors_every_level, dim=0)

        fam_assign_gt_ids_levels_batch = [ [] for _ in range(num_level)]
        fam_pos_targets_levels_batch = [ [] for _ in range(num_level)]
        if self.with_conf:
            
            bbox_pred_decode_every_level = p[6]
            bbox_pred_decode_batch_levels = [ [] for _ in range(batch_size)]  
            
            for batch_id in range(batch_size):
                for level_id in range(num_level):
                    
                    bbox_pred_decode = bbox_pred_decode_every_level[level_id][batch_id].reshape(-1,5)

                    bbox_pred_decode_batch_levels[batch_id].append(bbox_pred_decode)
            bbox_pred_decode_batch_levels = [torch.cat(i, dim=0) for i in bbox_pred_decode_batch_levels]

            conf_assign_gt_ids_levels_batch = [ [] for _ in range(num_level)]

            
            max_ious_levels_batch = [ [] for _ in range(num_level)]


        if self.with_alignconv:
            
            refine_anchors_levels = p[4]
            refine_anchors_batch_levels = [ [] for _ in range(batch_size)]  
            
            num_refine_anchors_every_level = [(refine_anchors_one_level.shape[1:3]).numel() for refine_anchors_one_level in refine_anchors_levels]        
            for batch_id in range(batch_size):
                for level_id in range(num_level):
                    
                    refine_anchors = refine_anchors_levels[level_id][batch_id].reshape(-1,5)
                    
                    refine_anchors_batch_levels[batch_id].append(refine_anchors)

            refine_anchors_batch_levels = [torch.cat(i, dim=0) for i in refine_anchors_batch_levels]
            odm_assign_gt_ids_levels_batch = [ [] for _ in range(num_level)]
            odm_pos_targets_levels_batch = [ [] for _ in range(num_level)]

        
        for batch_id in range(batch_size):
            targets_one_img = targets[targets[:,0]==batch_id].reshape(-1,7)            
            fam_assign_gt_ids, _ = assign_labels(init_grid_anchors_all_levels, targets_one_img[:, 2:7], imgs_size=self.imgs_size)

            fam_assign_gt_ids_levels = split_to_levels(fam_assign_gt_ids, num_grid_anchors_every_level)

            for level_id in range(num_level):
                
                fam_assign_gt_ids_one_level = fam_assign_gt_ids_levels[level_id]
                
                fam_assign_gt_ids_levels_batch[level_id].append(fam_assign_gt_ids_one_level)
        
                
                
                fam_pos_gt_ids_one_level = fam_assign_gt_ids_one_level >= 0
                
                fam_pos_targets_one_level = targets_one_img[fam_assign_gt_ids_one_level[fam_pos_gt_ids_one_level]].reshape(-1,7)
                fam_pos_targets_levels_batch[level_id].append(fam_pos_targets_one_level)
            

            
            if self.with_conf:
                bbox_pred_decode_one_img = bbox_pred_decode_batch_levels[batch_id]
                conf_assign_gt_ids, max_ious = assign_labels(bbox_pred_decode_one_img, targets_one_img[:, 2:7], imgs_size=self.imgs_size)
                
                conf_assign_gt_ids_levels = split_to_levels(conf_assign_gt_ids, num_grid_anchors_every_level)

                
                max_ious_levels = split_to_levels(max_ious, num_grid_anchors_every_level)
                for level_id in range(num_level):
                    
                    conf_assign_gt_ids_one_level = conf_assign_gt_ids_levels[level_id]
                    
                    conf_assign_gt_ids_levels_batch[level_id].append(conf_assign_gt_ids_one_level)

                    max_ious_levels_batch[level_id].append(max_ious_levels[level_id])
            
            
            if self.with_alignconv:
                '''然后进行odm模块的正负样本分配'''
                
                refine_anchor_one_img = refine_anchors_batch_levels[batch_id]
                
                odm_assign_gt_ids, _ = assign_labels(refine_anchor_one_img, targets_one_img[:, 2:7], imgs_size=self.imgs_size)

                
                odm_assign_gt_ids_levels = split_to_levels(odm_assign_gt_ids, num_refine_anchors_every_level)

                for level_id in range(num_level):
                    
                    odm_assign_gt_ids_one_level = odm_assign_gt_ids_levels[level_id]
                    
                    odm_assign_gt_ids_levels_batch[level_id].append(odm_assign_gt_ids_one_level)
            
                    
                    
                    odm_pos_gt_ids_one_level = odm_assign_gt_ids_one_level >= 0
                    
                    odm_pos_targets_one_level = targets_one_img[odm_assign_gt_ids_one_level[odm_pos_gt_ids_one_level]].reshape(-1,7)
                    odm_pos_targets_levels_batch[level_id].append(odm_pos_targets_one_level)


        
        
        fam_assign_gt_ids_levels = [torch.cat(i, dim=0) for i in fam_assign_gt_ids_levels_batch]
        
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
            
            conf_total_num_batchs_levels_pos = self.get_total_num_pos_batch_levels(conf_assign_gt_ids_levels, batch_size)

        odm_assign_gt_ids_levels, odm_pos_targets_levels, odm_total_num_batchs_levels_pos = None, None, None
        if self.with_alignconv:
            odm_assign_gt_ids_levels = [torch.cat(i, dim=0) for i in odm_assign_gt_ids_levels_batch]
            
            odm_total_num_batchs_levels_pos = self.get_total_num_pos_batch_levels(odm_assign_gt_ids_levels, batch_size)
            odm_pos_targets_levels = []
            for i in odm_pos_targets_levels_batch:
                if len(i):
                    odm_pos_targets_levels.append(torch.cat(i, dim=0))
                else:
                    odm_pos_targets_levels.append(None)

        return fam_assign_gt_ids_levels, fam_pos_targets_levels, fam_total_num_batchs_levels_pos, \
               odm_assign_gt_ids_levels, odm_pos_targets_levels, odm_total_num_batchs_levels_pos, max_ious_levels, conf_total_num_batchs_levels_pos   

    def get_total_num_pos_batch_levels(self, assign_gt_ids_levels, batch_size):
        total_num = sum((i>=0).sum().item() for i in assign_gt_ids_levels)
        total_num = max(total_num, batch_size)
        return total_num

    def compute_loss_single_level(self, bbox_pred, cls_pred, anchors, pos_targets_batch, assign_gt_ids_batch, module_name,conf_pred=None, max_ious_levels=None, index=None, odm_assign_gt_ids=None, odm_pos_targets=None):
        assert module_name in ("fam", "odm")

        device = anchors.device
        cls_loss, reg_loss = torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        pos_gt_ids_batch = assign_gt_ids_batch >= 0
        
        num_pos = pos_gt_ids_batch.sum().item()

        
        if pos_targets_batch is not None and num_pos>0:
            
            if module_name == 'fam':
 
                pos_anchors_batch = anchors[pos_gt_ids_batch]
                pos_reg_targets_batch = rboxes_encode(pos_anchors_batch, pos_targets_batch[:,2:7], is_encode_relative=self.is_encode_relative) 
                pos_bbox_pred_batch = bbox_pred[pos_gt_ids_batch]
                reg_loss = self.loss_fam_reg(pos_bbox_pred_batch, pos_reg_targets_batch) 

            if not self.with_alignconv:

                pos_cls_pred_batch = cls_pred[pos_gt_ids_batch]
                pos_cls_targets_batch = torch.zeros_like(pos_cls_pred_batch)  
                pos_cls_targets_batch[range(num_pos), pos_targets_batch[:,1].long()] = 1
                cls_loss = self.loss_fam_cls(pos_cls_pred_batch, pos_cls_targets_batch)

        else:
            pass
        
        if not self.with_alignconv:
            
            neg_gt_ids_batch = assign_gt_ids_batch == -1
            num_neg = neg_gt_ids_batch.sum().item()
            if num_neg>0:
                
                neg_cls_pred_batch = cls_pred[neg_gt_ids_batch]
                neg_cls_targets_batch = torch.zeros_like(neg_cls_pred_batch)
                cls_loss += self.loss_fam_cls(neg_cls_pred_batch, neg_cls_targets_batch)

        conf_loss = torch.zeros(1,device=device)
        if self.with_conf:
            conf_targets = max_ious_levels.detach()
            mask = (conf_targets >0) & (conf_targets < 1)
            if mask.sum()>0:

                conf_loss = self.loss_conf(conf_pred[mask], conf_targets[mask])
        if self.with_alignconv:
            pos_gt_ids_batch = odm_assign_gt_ids >= 0
            num_pos = pos_gt_ids_batch.sum().item()

            if odm_pos_targets is not None and num_pos>0:

                pos_cls_pred_batch = cls_pred[pos_gt_ids_batch]
                pos_cls_targets_batch = torch.zeros_like(pos_cls_pred_batch)
                pos_cls_targets_batch[range(num_pos), odm_pos_targets[:,1].long()] = 1
                cls_loss = self.loss_fam_cls(pos_cls_pred_batch, pos_cls_targets_batch)

            neg_gt_ids_batch = odm_assign_gt_ids == -1
            num_neg = neg_gt_ids_batch.sum().item()
            if num_neg>0:  
                neg_cls_pred_batch = cls_pred[neg_gt_ids_batch]
                neg_cls_targets_batch = torch.zeros_like(neg_cls_pred_batch)
                cls_loss += self.loss_fam_cls(neg_cls_pred_batch, neg_cls_targets_batch)

        return cls_loss, reg_loss, conf_loss

    def get_bboxes(self, p, module_name="fam"):

        assert module_name in ('fam', 'odm'), "must be FAM or ODM"

        batch_size = p[0][0].shape[0]
        num_level = len(p[0])
        cls_pred = p[0]
        bbox_pred_decode = p[6]        

        if self.with_conf:
            conf_pred = p[5]
        if self.with_alignconv:

            assert self.num_anchors > 1
            index_all = p[7]
            bbox_pred_decode = p[4]
            conf_choosed_pred = []
            for conf_pred_single_level, index_single_level in zip(conf_pred, index_all):
                
                _, feat_h, feat_w, _, _ = conf_pred_single_level.shape

                conf_choosed_pred.append(conf_pred_single_level[index_single_level].reshape(batch_size, feat_h, feat_w, -1))
                conf_pred = conf_choosed_pred
        imgs_results_ls = []
        for batch_id in range(batch_size):

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

            det_bboxes, det_labels = self.get_bboxes_single_img(scores_levels, bbox_pred_decode_levels)
            imgs_results_ls.append((det_bboxes, det_labels))
        
        return imgs_results_ls

    def get_bboxes_single_img(self, scores_levels, bbox_pred_decode_levels):

        max_before_nms_single_level = self.max_before_nms_per_level
        scores = []
        bboxes = []
        
        for score_level, bbox_pred_decode_level in zip(scores_levels, bbox_pred_decode_levels):
            if max_before_nms_single_level > 0 and score_level.shape[0] > max_before_nms_single_level:
                max_scores, _ = score_level.max(dim=1)
                _, topk_inds = max_scores.topk(max_before_nms_single_level)

                score_level = score_level[topk_inds, :] 
                bbox_pred_decode_level = bbox_pred_decode_level[topk_inds, :]     
            
            scores.append(score_level)
            bboxes.append(bbox_pred_decode_level)
        
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

        self.stride = [8, 16, 32]
        self.nl = len(self.stride)  
        self.backbone = DetectorBackbone(backbone_name)
        self.neck = PAN(
            in_channels=[512,1024,2048],
            num_outs=self.nl
        )

        self.head = RetinaNetHead(num_classes=num_classes, featmap_strides=self.stride)

    def forward(self, imgs, targets=None, post_process=False):
        outs = self.backbone(imgs)
        outs = self.neck(outs)
        if targets is not None:
            if not post_process:
                loss, loss_items = self.head(outs, targets=targets, imgs_size=imgs.shape[-2:], post_process=post_process)
                return loss, loss_items
            else:
                loss, loss_items, imgs_results_ls = self.head(outs, targets=targets, imgs_size=imgs.shape[-2:], post_process=post_process)
                return loss, loss_items, imgs_results_ls
        
        else:
            
            if post_process:
                imgs_results_ls = self.head(outs, post_process=post_process)
                return imgs_results_ls
            else:
                p = self.head(outs, post_process=post_process)
                return p

