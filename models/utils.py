import torch
from utils.metrics import bbox_iou_rotated
from functools import partial

from utils.general import rotated_box_to_poly_np

import torch.nn.functional as F


def multi_apply(func, *args, **kwargs):
    # 将函数func的参数kwargs固定，返回新的函数pfunc
    pfunc = partial(func, **kwargs) if kwargs else func
    # 这里的args表示feats和anchor_strides两个序列，map函数会分别遍历这两个序列，然后送入pfunc函数
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def split_to_levels(assign_gt_ids, num_anchors_every_level):
    '''
      将所有特征层级的正负样本分配结果，拆分为一个列表，存储各个特征层级的分配结果

    assign_gt_ids : shape:(all_levels_anchors)
    num_anchors_every_level: 每一个特征层级的网格anchors的个数
    '''

    level_assign_gt_ids = []
    start = 0
    for n in num_anchors_every_level:
        end = start + n
        level_assign_gt_ids.append(assign_gt_ids[start:end].squeeze(0))
        start = end
    
    # 是个列表，存储每一个特征层级的正负样本分配结果
    return level_assign_gt_ids


# def assign_labels(anchors, gt_boxes, imgs_size=(1024,1024), 
#         pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou_thr=0, gt_max_assign_all=True,
#         filter_invalid_anchors=False, filter_invalid_ious=True
#     ):
#     '''
#     anchors shape : [M, 5], x y w h 都是以像素为单位的，角度为弧度，并且不是归一化值
#     gt_boxes shape : [N, 5]

#     min_pos_iou_thr: 对于那些与所有anchor的IOU都小于正样本阈值的gt，分配最大IOU的anchor来预测这个gt，
#                     但是这个最大IOU也不能太小，于是设置这个阈值
#     gt_max_assign_all: 上述情况下，gt可能与多个anchor都具有最大IOU，此时就把多个anchor都设置为正样本
#     filter_invalid_anchors: invalid_anchor是指参数超出图像边界的anchor，无效anchors设置为忽略样本
#     filter_invalid_ious ：旋转iou的计算函数有问题，范围不在[0,1.0]之间，将超出范围的iou设置为负值，从而成为忽略样本
#     return:
#         assign_gt_ids : [M], -2表示是忽略样本，-1表示是负样本，大于等于0表示是正样本，具体的数值表示是预测的哪个gt
#     '''


#     # 必须保证传入bbox_iou_rotated的参数的内存连续性
#     # 否则不会报错，但会出现计算错误，这种bug更难排查
#     if not anchors.is_contiguous():
#         anchors = anchors.contiguous()
#     if not gt_boxes.is_contiguous():
#         gt_boxes = gt_boxes.contiguous()

#     device = gt_boxes.device

#     num_anchors = anchors.shape[0]
#     num_gt_boxes = gt_boxes.shape[0]

#     # 标签分配的结果，一维向量，每个位置表示一个anchor，其值表示该anchor预测哪一个gt，默认全是忽略样本
#     assign_gt_ids = torch.ones(num_anchors, dtype=torch.long, device=device)*(-2)

#     # ## 判断anchor是否有超出图像边界的情况
#     # # 对于init_anchors没有这种问题，但对于refine_anchors可能存在超出边界的问题
#     if filter_invalid_anchors:
#         flags = (anchors[:, 0] >= 0) & \
#                 (anchors[:, 1] >= 0) & \
#                 (anchors[:, 2] < imgs_size[1]) & \
#                 (anchors[:, 3] < imgs_size[0])    
    
#     # 首先判断gt_boxes是否为空
#     if num_gt_boxes == 0:
#         # 如果为空，那么anchors就只有可能是负样本，或者因为anchors无效而成为忽略样本
#         # 将有效的anchors设置为负样本
#         if filter_invalid_anchors:
#             assign_gt_ids[flags] = -1
#         else:
#             assign_gt_ids[:] = -1

#         return assign_gt_ids

#     ious = bbox_iou_rotated(anchors, gt_boxes)

#     if filter_invalid_ious:
#         # assert torch.all((ious>=0) & (ious<=1))
#         # "iou 的值应该大于等于0，并且小于等于1"
#         if not torch.all((ious>=0) & (ious<=1)):
#             inds = (ious<0) | (ious>1)
#             # print(ious[inds])
#             # # 设置为负值, 则可以保证这一对iou值不会被标注为正样本或负样本
#             ious[inds] = -0.5

#         #     # print(f"处于{mode}模块下，有几个iou计算不准确")   
    
#     # 无效的anchors的iou设置为负值，使该anchors成为忽略样本
#     if filter_invalid_anchors:
#         ious[~flags] = -0.5

#     # # 打印一下iou比较大的anchors和gt
#     # if torch.any(ious > 0.98):
#     #     inds_y, inds_x = torch.where(ious > 0.98)

#     #     print("iou>0.98")
#     #     print("anchors:", anchors[inds_y])
#     #     print("gt:", gt_boxes[inds_x])

#     # 下面对各种可能存在的情况进行处理
#     # 1.首先分配负样本，负样本的定义，与所有的gt的IOU都小于负样本阈值的anchor，也就是最大值都小于负样本阈值
#     # 如果最大值有2个，.max()函数只会返回第一个最大值的坐标索引
#     max_ious, argmax_ious = ious.max(dim=1)
#     assign_gt_ids[(max_ious>=0)&(max_ious<neg_iou_thr)] = -1
    


#     # 2.然后分配正样本，正样本有两种定义
#     # （1）正样本1：一个anchor与某些gt的IOU大于pos_iou_thr：可能与多个gt的IOU都大于阈值，只选择最大IOU的gt分配给该anchor
#     # 如果最大IOU值有2个，.max()函数只会返回第一个最大值的坐标索引，因此这里我们会忽略其它的gt框
#     pos_inds = max_ious >= pos_iou_thr
#     assign_gt_ids[pos_inds] = argmax_ious[pos_inds]


#     # （2）正样本2：一个gt与所有anchor的IOU都小于正样本阈值，但是又不能不预测这个gt，所以把他分配给具有最大IOU的anchor，这个anchor也是正样本
#     # 有可能具有最大IOU阈值的anchor有多个，那么这些anchor都可以作为正样本
#     # 如果最大值有2个，.max()函数只会返回第一个最大值的坐标索引
#     gt_max_ious, gt_argmax_ious = ious.max(dim=0)
#     # 逐个gt框进行处理
#     for i in range(num_gt_boxes):
#         # 我认为这个地方设置为大于更合适，因为如果是>=，且min_pos_iou_thr=0.0，那么可能iou=0的也设置为正样本了
#         if gt_max_ious[i] > min_pos_iou_thr:
#             if gt_max_assign_all:
#                 # 有可能具有最大IOU阈值的anchor有多个，那么这些anchor都可以作为正样本
#                 # 找到具有最大iou的anchor
#                 max_iou_ids = ious[:,i] == gt_max_ious[i]
#                 # max_iou_ids = (ious[:,i] > 0) & (ious[:,i] > gt_max_ious[i] - 1e-2)
#                 # 将该gt分配给这些anchor
#                 assign_gt_ids[max_iou_ids] = i

#             else:
#                 assign_gt_ids[gt_argmax_ious[i]] = i

    
#     return assign_gt_ids



def assign_labels(anchors, gt_boxes, imgs_size=(1024,1024), 
        pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou_thr=0, gt_max_assign_all=True,
        filter_invalid_anchors=False, filter_invalid_ious=True
    ):
    '''
    anchors shape : [M, 5], x y w h 都是以像素为单位的，角度为弧度，并且不是归一化值
    gt_boxes shape : [N, 5]

    min_pos_iou_thr: 对于那些与所有anchor的IOU都小于正样本阈值的gt，分配最大IOU的anchor来预测这个gt，
                    但是这个最大IOU也不能太小，于是设置这个阈值
    gt_max_assign_all: 上述情况下，gt可能与多个anchor都具有最大IOU，此时就把多个anchor都设置为正样本
    filter_invalid_anchors: invalid_anchor是指参数超出图像边界的anchor，无效anchors设置为忽略样本
    filter_invalid_ious ：旋转iou的计算函数有问题，范围不在[0,1.0]之间，将超出范围的iou设置为负值，从而成为忽略样本
    return:
        assign_gt_ids : [M], -2表示是忽略样本，-1表示是负样本，大于等于0表示是正样本，具体的数值表示是预测的哪个gt
    '''


    # 必须保证传入bbox_iou_rotated的参数的内存连续性
    # 否则不会报错，但会出现计算错误，这种bug更难排查
    if not anchors.is_contiguous():
        anchors = anchors.contiguous()
    if not gt_boxes.is_contiguous():
        gt_boxes = gt_boxes.contiguous()

    device = gt_boxes.device

    num_anchors = anchors.shape[0]
    num_gt_boxes = gt_boxes.shape[0]

    # 标签分配的结果，一维向量，每个位置表示一个anchor，其值表示该anchor预测哪一个gt，默认全是忽略样本
    assign_gt_ids = torch.ones(num_anchors, dtype=torch.long, device=device)*(-2)

    # ## 判断anchor是否有超出图像边界的情况
    # # 对于init_anchors没有这种问题，但对于refine_anchors可能存在超出边界的问题
    if filter_invalid_anchors:
        flags = (anchors[:, 0] >= 0) & \
                (anchors[:, 1] >= 0) & \
                (anchors[:, 2] < imgs_size[1]) & \
                (anchors[:, 3] < imgs_size[0])    
    
    # 首先判断gt_boxes是否为空
    if num_gt_boxes == 0:
        # 如果为空，那么anchors就只有可能是负样本，或者因为anchors无效而成为忽略样本
        # 将有效的anchors设置为负样本
        if filter_invalid_anchors:
            assign_gt_ids[flags] = -1
        else:
            assign_gt_ids[:] = -1

        max_ious = torch.zeros(num_anchors, dtype=torch.float32, device=device)
        return assign_gt_ids, max_ious

    ious = bbox_iou_rotated(anchors, gt_boxes)

    if filter_invalid_ious:
        # assert torch.all((ious>=0) & (ious<=1))
        # "iou 的值应该大于等于0，并且小于等于1"
        if not torch.all((ious>=0) & (ious<=1)):
            inds = (ious<0) | (ious>1)
            # print(ious[inds])
            # # 设置为负值, 则可以保证这一对iou值不会被标注为正样本或负样本
            ious[inds] = -0.5

        #     # print(f"处于{mode}模块下，有几个iou计算不准确")   
    
    # 无效的anchors的iou设置为负值，使该anchors成为忽略样本
    if filter_invalid_anchors:
        ious[~flags] = -0.5

    # # 打印一下iou比较大的anchors和gt
    # if torch.any(ious > 0.98):
    #     inds_y, inds_x = torch.where(ious > 0.98)

    #     print("iou>0.98")
    #     print("anchors:", anchors[inds_y])
    #     print("gt:", gt_boxes[inds_x])

    # 下面对各种可能存在的情况进行处理
    # 1.首先分配负样本，负样本的定义，与所有的gt的IOU都小于负样本阈值的anchor，也就是最大值都小于负样本阈值
    # 如果最大值有2个，.max()函数只会返回第一个最大值的坐标索引
    max_ious, argmax_ious = ious.max(dim=1)
    assign_gt_ids[(max_ious>=0)&(max_ious<neg_iou_thr)] = -1
    


    # 2.然后分配正样本，正样本有两种定义
    # （1）正样本1：一个anchor与某些gt的IOU大于pos_iou_thr：可能与多个gt的IOU都大于阈值，只选择最大IOU的gt分配给该anchor
    # 如果最大IOU值有2个，.max()函数只会返回第一个最大值的坐标索引，因此这里我们会忽略其它的gt框
    pos_inds = max_ious >= pos_iou_thr
    assign_gt_ids[pos_inds] = argmax_ious[pos_inds]


    # （2）正样本2：一个gt与所有anchor的IOU都小于正样本阈值，但是又不能不预测这个gt，所以把他分配给具有最大IOU的anchor，这个anchor也是正样本
    # 有可能具有最大IOU阈值的anchor有多个，那么这些anchor都可以作为正样本
    # 如果最大值有2个，.max()函数只会返回第一个最大值的坐标索引
    gt_max_ious, gt_argmax_ious = ious.max(dim=0)
    # 逐个gt框进行处理
    for i in range(num_gt_boxes):
        # 我认为这个地方设置为大于更合适，因为如果是>=，且min_pos_iou_thr=0.0，那么可能iou=0的也设置为正样本了
        if gt_max_ious[i] > min_pos_iou_thr:
            if gt_max_assign_all:
                # 有可能具有最大IOU阈值的anchor有多个，那么这些anchor都可以作为正样本
                # 找到具有最大iou的anchor
                max_iou_ids = ious[:,i] == gt_max_ious[i]
                # max_iou_ids = (ious[:,i] > 0) & (ious[:,i] > gt_max_ious[i] - 1e-2)
                # 将该gt分配给这些anchor
                assign_gt_ids[max_iou_ids] = i

            else:
                assign_gt_ids[gt_argmax_ious[i]] = i

    # # 每个anchor都有一个iou，这个iou作为conf分支的回归标签
    # max_ious[max_ious<0] = 0
    # max_ious[max_ious>1] = 1
    
    return assign_gt_ids, max_ious

# 判断网格点是否落在了gt-boxes内部或gt-boxes中心点的center_radius*stride区域
def get_in_boxes_info(pred_boxes, gt_boxes, grid_xy_center, grid_stride, center_radius=2.5):
    '''
        pred_boxes : shape [num_grids, 5]
        grid_xy_center : shape [num_grids, 2]
    '''

    num_gts = gt_boxes.shape[0]
    num_grids = pred_boxes.shape[0]
    # shape [n_grids] -> [n_gt, n_grids]
    grid_x_center = grid_xy_center[:, 0].reshape(1, -1).repeat(num_gts, 1)
    grid_y_center = grid_xy_center[:, 1].reshape(1, -1).repeat(num_gts, 1)

    # # 计算gt框的xmin,ymin,xmax,ymax，shape为[num_gt, total_num_anchors]
    # shape [num_gts, 8]
    gt_boxes_points = torch.from_numpy(rotated_box_to_poly_np(gt_boxes.cpu().numpy())).to(gt_boxes.device)
    # shape为[num_gt, n_grids]
    gt_boxes_l = gt_boxes_points[:, 0::2]    # 从0开始，每隔2个取一个，取到最后，即取索引0、2、4......
    gt_boxes_l = gt_boxes_l.min(dim=1).values.reshape(-1,1).repeat(1, num_grids)
    gt_boxes_r = gt_boxes_points[:, 0::2]    # 从0开始，每隔2个取一个，取到最后，即取索引0、2、4......
    gt_boxes_r = gt_boxes_r.max(dim=1).values.reshape(-1,1).repeat(1, num_grids)
    gt_boxes_t = gt_boxes_points[:, 1::2]    # 从0开始，每隔2个取一个，取到最后，即取索引0、2、4......
    gt_boxes_t = gt_boxes_t.min(dim=1).values.reshape(-1,1).repeat(1, num_grids)
    gt_boxes_b = gt_boxes_points[:, 1::2]    # 从0开始，每隔2个取一个，取到最后，即取索引0、2、4......
    gt_boxes_b = gt_boxes_b.max(dim=1).values.reshape(-1,1).repeat(1, num_grids)

    # 网格中心点坐标与gt的边界作差，判断网格中心点是否落在该gt框内部
    b_l = grid_x_center - gt_boxes_l
    b_r = gt_boxes_r - grid_x_center
    b_t = grid_y_center - gt_boxes_t
    b_b = gt_boxes_b - grid_y_center

    # shape为[num_gt, n_grids, 4]
    bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

    # is_in_boxes shape [num_gt, n_grids]，网格中心点是否落在gt内部
    is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
    # is_in_boxes_all shape [n_grids]，即该网格中心点是否落在了一个或多个gt的内部
    is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
    
    
    # 判断网格点是否落在了gt的中心区域
    # in fixed center
    gt_boxes_l = gt_boxes[:, 0].reshape(-1,1).repeat(1, num_grids) - center_radius * grid_stride.reshape(1,-1)
    gt_boxes_r = gt_boxes[:, 0].reshape(-1,1).repeat(1, num_grids) + center_radius * grid_stride.reshape(1,-1)
    gt_boxes_t = gt_boxes[:, 1].reshape(-1,1).repeat(1, num_grids) - center_radius * grid_stride.reshape(1,-1)
    gt_boxes_b = gt_boxes[:, 1].reshape(-1,1).repeat(1, num_grids) + center_radius * grid_stride.reshape(1,-1)

    c_l = grid_x_center - gt_boxes_l
    c_r = gt_boxes_r - grid_x_center
    c_t = grid_y_center - gt_boxes_t
    c_b = gt_boxes_b - grid_y_center
    center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
    is_in_centers = center_deltas.min(dim=-1).values > 0.0
    is_in_centers_all = is_in_centers.sum(dim=0) > 0


    # 网格中心点，或者落在gt内部，或者落在gt中心点的center_radius*stride区域内部
    # in boxes and in centers
    # shape [num_grid_all_levels]
    is_in_boxes_or_center = is_in_boxes_all | is_in_centers_all

    # is_in_boxes_and_center表示既在boxes内部、又在boxes center内部的网格中心点
    # shape [num_gt, num_filter_grids]
    is_in_boxes_and_center = (
        is_in_boxes[:, is_in_boxes_or_center] & is_in_centers[:, is_in_boxes_or_center]
    )
    return is_in_boxes_or_center, is_in_boxes_and_center



def dynamic_k_matching(cost, pair_wise_ious, num_gt, fg_mask, num_topk=10):
    # Dynamic K
    # ---------------------------------------------------------------
    
    # cost shape [num_gt, num_in_boxes_or_center]
    matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

    # iou，shape [num_gt, num_in_boxes_or_center]
    # 根据iou的大小，每个gt选择top-k个候选框，k=10，如果候选框不足10，则全部选择出来
    ious_in_boxes_matrix = pair_wise_ious
    n_candidate_k = min(num_topk, ious_in_boxes_matrix.size(1))
    topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
    
    # dynamic-k，把top-k的iou全部加起来，求和取整数，作为每个gt的dynamic-k
    dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
    dynamic_ks = dynamic_ks.tolist()
    for gt_idx in range(num_gt):
        # largest=False，表示返回最小值，即从小到大排序，然后取top-k
        _, pos_idx = torch.topk(
            cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
        )
        # 对应位置设置为正样本
        matching_matrix[gt_idx][pos_idx] = 1

    del topk_ious, dynamic_ks, pos_idx

    # shape [num_in_boxes_or_center]，每个anchor的匹配的gt个数
    anchor_matching_gt = matching_matrix.sum(0)
    # 如果有一个或多个anchor，匹配了2个及以上的gt
    if (anchor_matching_gt > 1).sum() > 0:
        # 找到匹配了2个及以上gt的anchor，找到其最小cost的gt
        _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
        # 将匹配多个gt的anchor的匹配全部设置为0
        matching_matrix[:, anchor_matching_gt > 1] *= 0
        # 将该anchor具有最小cost的gt设置为匹配
        matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
    
    # 前景mask，在num_in_boxes_or_center中选择dynamic-k个正样本，shape [num_in_boxes_or_center]
    fg_mask_inboxes = matching_matrix.sum(0) > 0
    num_fg = fg_mask_inboxes.sum().item()

    # # 对fg_mask进行了更新
    # # fg_mask shape [num_grid_all_levels]
    # fg_mask[fg_mask.clone()] = fg_mask_inboxes

    # matching_matrix[:, fg_mask_inboxes] shape为[num_gt, num_filter_anchor]
    # matched_gt_inds shape [num_filter_anchor]，找到匹配的gt的索引
    matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
    # gt_matched_classes = gt_classes[matched_gt_inds]

    # # (matching_matrix * pair_wise_ious).sum(0) shape [num_in_boxes_anchor]
    # pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
    #     fg_mask_inboxes
    # ]

    # 标签分配的结果，一维向量，每个位置表示一个anchor，其值表示该anchor预测哪一个gt，默认全是负样本
    # 需要注意的是，这种标签分配方法没有忽略样本
    assign_gt_ids = torch.ones(fg_mask.shape[0], dtype=torch.long, device=fg_mask.device)*(-1)

    # # 这种方法无法为assign_gt_ids赋值，经过了两次索引
    # assign_gt_ids[fg_mask][fg_mask_inboxes] = matched_gt_inds
    # 改成下面这种方法
    assign_gt_ids_temp = torch.ones(fg_mask_inboxes.shape[0], dtype=torch.long, device=fg_mask.device)*(-1)
    assign_gt_ids_temp[fg_mask_inboxes] = matched_gt_inds
    assign_gt_ids[fg_mask] = assign_gt_ids_temp
    
    # print(f"assign_gt_ids正样本个数：{(assign_gt_ids>=0).sum()}")
    # print(f"matched_gt_inds正样本个数：{(matched_gt_inds>=0).sum()}")

    return assign_gt_ids


# YOLOX中的动态标签分配法
def assign_labels_SimOTA(pred_boxes, targets_one_img, cls_pred, grid_xy_center, grid_stride, imgs_size=(1024,1024), 
        pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou_thr=0, gt_max_assign_all=True,
        filter_invalid_anchors=False, filter_invalid_ious=True
    ):
    '''
    pred_boxes shape : [M, 5], x y w h 都是以像素为单位的，角度为弧度，并且不是归一化值
    gt_boxes shape : [N, 5]

    min_pos_iou_thr: 对于那些与所有anchor的IOU都小于正样本阈值的gt，分配最大IOU的anchor来预测这个gt，
                    但是这个最大IOU也不能太小，于是设置这个阈值
    gt_max_assign_all: 上述情况下，gt可能与多个anchor都具有最大IOU，此时就把多个anchor都设置为正样本
    filter_invalid_anchors: invalid_anchor是指参数超出图像边界的anchor，无效anchors设置为忽略样本
    filter_invalid_ious ：旋转iou的计算函数有问题，范围不在[0,1.0]之间，将超出范围的iou设置为负值，从而成为忽略样本
    return:
        assign_gt_ids : [M], -2表示是忽略样本，-1表示是负样本，大于等于0表示是正样本，具体的数值表示是预测的哪个gt
    '''

    # pred_boxes shape ()
    assert targets_one_img.shape[1] == 6
    gt_boxes = targets_one_img[:, 1:6]
    gt_classes = targets_one_img[:, 0].reshape(-1)

    # 必须保证传入bbox_iou_rotated的参数的内存连续性
    # 否则不会报错，但会出现计算错误，这种bug更难排查
    if not pred_boxes.is_contiguous():
        pred_boxes = pred_boxes.contiguous()
    if not gt_boxes.is_contiguous():
        gt_boxes = gt_boxes.contiguous()

    device = gt_boxes.device

    num_anchors = pred_boxes.shape[0]
    num_gt = gt_boxes.shape[0]


    # 标签分配的结果，一维向量，每个位置表示一个anchor，其值表示该anchor预测哪一个gt，默认全是负样本
    # 需要注意的是，这种标签分配方法没有忽略样本
    assign_gt_ids = torch.ones(num_anchors, dtype=torch.long, device=device)*(-1)
    # 首先判断gt_boxes是否为空
    if num_gt == 0:
        return assign_gt_ids
    


    # 找到落在gt内部或中心区域的网格点，即中心区域的网格点
    in_boxes_or_center_mask, is_in_boxes_and_center = get_in_boxes_info(pred_boxes.clone(), gt_boxes.clone(), grid_xy_center, grid_stride)

    ## 下面根据cost去计算dynamic-k
    # 取出中心区域的预测框
    pred_boxes = pred_boxes[in_boxes_or_center_mask]
    # 中心区域网格的个数
    num_center_samples = pred_boxes.shape[0]

    ## 计算IOU损失
    # shape [num_gt, num_center_samples]
    pair_wise_ious = bbox_iou_rotated(gt_boxes, pred_boxes)
    if filter_invalid_ious:
        # assert torch.all((ious>=0) & (ious<=1))
        # "iou 的值应该大于等于0，并且小于等于1"
        if not torch.all((pair_wise_ious>=0) & (pair_wise_ious<=1)):
            inds = (pair_wise_ious<0) | (pair_wise_ious>1)
            # print(ious[inds])
            # # 设置为负值, 则可以保证这一对iou值不会被标注为正样本或负样本
            pair_wise_ious[inds] = 0.0
    pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
    
    ## 计算分类损失
    num_classes = cls_pred.shape[1]
    # shape [num_center_samples, num_classes]
    cls_pred = cls_pred[in_boxes_or_center_mask]
    # shape [num_gt, num_center_samples, num_classes]
    cls_pred = cls_pred.unsqueeze(0).repeat(num_gt, 1, 1)
    # 计算分类的回归标签
    # shape [num_gt, num_center_samples, num_classes]
    gt_cls_targets = F.one_hot(gt_classes.to(torch.int64), num_classes).float().unsqueeze(1).repeat(1, num_center_samples, 1)

    pair_wise_cls_loss = F.binary_cross_entropy_with_logits(cls_pred, gt_cls_targets, reduction="none").sum(-1)

    # 用于挑选dynamic-k的cost
    # shape [num_gt, num_in_boxes_anchor]，is_in_boxes_and_center取反，只在boxes内部、只在boxes中心区域内部和两者都不在的网格的cost特别大
    cost = (
        pair_wise_cls_loss
        + 3.0 * pair_wise_ious_loss
        + 100000.0 * (~is_in_boxes_and_center)
    )


    # 标签分配的结果，一维向量，每个位置表示一个anchor，其值表示该anchor预测哪一个gt，默认全是负样本，没有忽略样本
    assign_gt_ids = dynamic_k_matching(cost, pair_wise_ious, num_gt, in_boxes_or_center_mask.clone())

    # # 打印一下iou比较大的anchors和gt
    # if torch.any(ious > 0.98):
    #     inds_y, inds_x = torch.where(ious > 0.98)

    #     print("iou>0.98")
    #     print("anchors:", anchors[inds_y])
    #     print("gt:", gt_boxes[inds_x])

    
    return assign_gt_ids
