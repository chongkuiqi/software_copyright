# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""


from .box_iou_rotated import box_iou_rotated


def bbox_iou_rotated(rboxes1, rboxes2):
    """Calculate IoU between 2D bboxes.

    Args:
        bboxes1 (Tensor): bboxes have shape (m, 5) in <x, y, w, h, a>
            format, or shape (m, 5) in <x, y, w, h, a, score> format.
        bboxes2 (Tensor): bboxes have shape (m, 5) in <x, y, w, h, a>
            format, or shape (m, 5) in <x, y, w, h, a, score> format, or be
            empty. 
        # bboxes1和bboxes2可以是cpu张量，也可以是cuda张量，但必须是同一个设备上的
    Returns:
        Tensor: shape (m, n) 
    """
    assert rboxes1.size(-1) in [0, 5, 6]
    assert rboxes2.size(-1) in [0, 5, 6]
    if rboxes2.size(-1) == 6:
        rboxes2 = rboxes2[..., :5]
    if rboxes1.size(-1) == 6:
        rboxes1 = rboxes1[..., :5]

    ious = box_iou_rotated(rboxes1.float(), rboxes2.float())
    
    return ious

