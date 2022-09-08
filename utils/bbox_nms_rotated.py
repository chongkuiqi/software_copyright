import torch
from .ml_nms_rotated import ml_nms_rotated

# import torch.tensor as Tensor
# from typing import Optional


def multiclass_nms_rotated(bboxes,
                           scores,
                           score_thr=0.05,
                           iou_thr=0.5,
                           max_per_img = 2000):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, 5)
        multi_scores (Tensor): shape (n, num_classes), don't have the background class
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): NMS IoU threshold
        max_per_img (int): if there are more than max_per_img bboxes after NMS,
            only top max_per_img will be kept.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 6) and (k, 1). Labels
            are 0-based.
    """
    num_classes = scores.size(1)

    # 增加边界框的个数，即一个特征位置点上，共有num_classes个框，代表不同类别的框
    assert bboxes.shape[1] == 5
    # bboxes shape:[N, num_classes, 5]
    bboxes = bboxes[:, None].expand(-1, num_classes, 5)

    ## 根据类别分数进行阈值过滤
    # score shape [N,num_classes]
    mask = scores > score_thr
    # scores shape [N2]
    scores = scores[mask]
    # bboxes shape [N2, 5]
    bboxes = bboxes[mask]

    # .nonzero()返回非零位置的横纵坐标
    labels = mask.nonzero(as_tuple=False)[:, 1]
    # 放在同一个设备上
    labels = labels.to(bboxes)

    
    if bboxes.shape[0] > 0:
        # 进行多类别的NMS
        keep = ml_nms_rotated(bboxes, scores, labels, iou_thr)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if keep.size(0) > max_per_img:
            _, inds = scores.sort(descending=True)
            inds = inds[:max_per_img]
            bboxes = bboxes[inds]
            scores = scores[inds]
            labels = labels[inds]
        return torch.cat([bboxes, scores[:, None]], dim=1), labels
    # 如果进行上述处理后，没有目标
    else:
        # boxes shape为[N,6], x,y,w,h,theta,score
        bboxes = bboxes.new_zeros((0, 6))
        labels = bboxes.new_zeros((0,1), dtype=torch.long)
        return bboxes, labels


# def batched_nms(boxes: Tensor,
#                 scores: Tensor,
#                 idxs: Tensor,
#                 nms_cfg: Optional[Dict],
#                 class_agnostic: bool = False):
#     r"""Performs non-maximum suppression in a batched fashion.

#     Modified from `torchvision/ops/boxes.py#L39
#     <https://github.com/pytorch/vision/blob/
#     505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39>`_.
#     In order to perform NMS independently per class, we add an offset to all
#     the boxes. The offset is dependent only on the class idx, and is large
#     enough so that boxes from different classes do not overlap.

#     Note:
#         In v1.4.1 and later, ``batched_nms`` supports skipping the NMS and
#         returns sorted raw results when `nms_cfg` is None.

#     Args:
#         boxes (torch.Tensor): boxes in shape (N, 4) or (N, 5).
#         scores (torch.Tensor): scores in shape (N, ).
#         idxs (torch.Tensor): each index value correspond to a bbox cluster,
#             and NMS will not be applied between elements of different idxs,
#             shape (N, ).
#         nms_cfg (dict | optional): Supports skipping the nms when `nms_cfg`
#             is None, otherwise it should specify nms type and other
#             parameters like `iou_thr`. Possible keys includes the following.

#             - iou_threshold (float): IoU threshold used for NMS.
#             - split_thr (float): threshold number of boxes. In some cases the
#               number of boxes is large (e.g., 200k). To avoid OOM during
#               training, the users could set `split_thr` to a small value.
#               If the number of boxes is greater than the threshold, it will
#               perform NMS on each group of boxes separately and sequentially.
#               Defaults to 10000.
#         class_agnostic (bool): if true, nms is class agnostic,
#             i.e. IoU thresholding happens over all boxes,
#             regardless of the predicted class. Defaults to False.

#     Returns:
#         tuple: kept dets and indice.  -> Tuple[Tensor, Tensor]

#         - boxes (Tensor): Bboxes with score after nms, has shape
#           (num_bboxes, 5). last dimension 5 arrange as
#           (x1, y1, x2, y2, score)
#         - keep (Tensor): The indices of remaining boxes in input
#           boxes.
#     """
#     # skip nms when nms_cfg is None
#     if nms_cfg is None:
#         scores, inds = scores.sort(descending=True)
#         boxes = boxes[inds]
#         return torch.cat([boxes, scores[:, None]], -1), inds

#     nms_cfg_ = nms_cfg.copy()
#     class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
#     if class_agnostic:
#         boxes_for_nms = boxes
#     else:
#         # When using rotated boxes, only apply offsets on center.
#         if boxes.size(-1) == 5:
#             # Strictly, the maximum coordinates of the rotating box
#             # (x,y,w,h,a) should be calculated by polygon coordinates.
#             # But the conversion from rotated box to polygon will
#             # slow down the speed.
#             # So we use max(x,y) + max(w,h) as max coordinate
#             # which is larger than polygon max coordinate
#             # max(x1, y1, x2, y2,x3, y3, x4, y4)
#             max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
#             offsets = idxs.to(boxes) * (
#                 max_coordinate + torch.tensor(1).to(boxes))
#             boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
#             boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]],
#                                       dim=-1)
#         else:
#             max_coordinate = boxes.max()
#             offsets = idxs.to(boxes) * (
#                 max_coordinate + torch.tensor(1).to(boxes))
#             boxes_for_nms = boxes + offsets[:, None]

#     nms_type = nms_cfg_.pop('type', 'nms')
#     nms_op = eval(nms_type)

#     split_thr = nms_cfg_.pop('split_thr', 10000)
#     # Won't split to multiple nms nodes when exporting to onnx
#     if boxes_for_nms.shape[0] < split_thr or torch.onnx.is_in_onnx_export():
#         dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
#         boxes = boxes[keep]

#         # This assumes `dets` has arbitrary dimensions where
#         # the last dimension is score.
#         # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
#         # rotated boxes [cx, cy, w, h, angle_radian, score].

#         scores = dets[:, -1]
#     else:
#         max_num = nms_cfg_.pop('max_num', -1)
#         total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
#         # Some type of nms would reweight the score, such as SoftNMS
#         scores_after_nms = scores.new_zeros(scores.size())
#         for id in torch.unique(idxs):
#             mask = (idxs == id).nonzero(as_tuple=False).view(-1)
#             dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
#             total_mask[mask[keep]] = True
#             scores_after_nms[mask[keep]] = dets[:, -1]
#         keep = total_mask.nonzero(as_tuple=False).view(-1)

#         scores, inds = scores_after_nms[keep].sort(descending=True)
#         keep = keep[inds]
#         boxes = boxes[keep]

#         if max_num > 0:
#             keep = keep[:max_num]
#             boxes = boxes[:max_num]
#             scores = scores[:max_num]

#     boxes = torch.cat([boxes, scores[:, None]], -1)
#     return boxes, keep
