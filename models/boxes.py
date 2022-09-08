import numpy as np
import torch
from utils.general import norm_angle

def norm_angle(angle):
    range = [-np.pi / 4, np.pi]
    # range = [0, np.pi]

    # 对pi取余数，获得超过pi的部分，然后在
    ang = (angle - range[0]) % range[1] + range[0]
    return ang


# 对旋转框进行编码
def rboxes_encode(anchors, gt_rboxes, is_encode_relative=True):
    '''Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            anchors (torch.Tensor): [N,5], x,y,w,h,theta, 单位为像素值, theta为弧度
            gt_bboxes (torch.Tensor): [N,5], x,y,w,h,theta, 单位为像素值, theta为弧度

        Returns:
            encoded_rboxes(torch.Tensor): Box transformation deltas
        
    '''
    # 保证anchor的个数个gt的个数一样
    assert anchors.size(0) == gt_rboxes.size(0)
    assert anchors.size(-1) == gt_rboxes.size(-1) == 5

    # 下面进行编码
    gt_w = gt_rboxes[..., 2]
    gt_h = gt_rboxes[..., 3]
    gt_angle = gt_rboxes[..., 4]

    anchors_w = anchors[..., 2]
    anchors_h = anchors[..., 3]
    anchors_angles = anchors[..., 4]

    # gt与anchor的中心点坐标的offsets
    xy_offsets = gt_rboxes[..., 0:2] - anchors[..., 0:2]

    ## 计算offsets
    # 由于是方形anchor，且角度为0，所以anchors_w确实是水平方向的长度，可以用来标准化，但是无法确保归一化
    # 此外，这里的xy的offsets的计算不太理解，不知道为什么角度参与计算；
    # 杨学的R3det算法等都不是采用的这种方法，而ROITransformer等就采用的类似的方法
    if is_encode_relative:
        cosa = torch.cos(anchors_angles)
        sina = torch.sin(anchors_angles)
        dx = (cosa * xy_offsets[..., 0] + sina * xy_offsets[..., 1]) / anchors_w
        dy = (-sina * xy_offsets[..., 0] + cosa * xy_offsets[..., 1]) / anchors_h
    else:
        dx = xy_offsets[..., 0] / anchors_w
        dy = xy_offsets[..., 1] / anchors_h
    # torch.log是以e为底的，标准化，但是无法确保归一化
    dw = torch.log(gt_w / anchors_w)
    dh = torch.log(gt_h / anchors_h)
    
    # 角度差转化到[-0.25pi,0.75pi]之间，然后进行归一化
    da = (gt_angle - anchors_angles)
    da = norm_angle(da) / np.pi

    encoded_rboxes = torch.stack((dx, dy, dw, dh, da), -1).reshape(-1,5)

    # means = deltas.new_tensor(means).unsqueeze(0)
    # stds = deltas.new_tensor(stds).unsqueeze(0)
    # deltas = deltas.sub_(means).div_(stds)

    return encoded_rboxes


def rboxes_decode(anchors,
               pred_bboxes,
               is_encode_relative=True,
               wh_ratio_clip=16 / 1000 ):
    """Apply transformation `pred_bboxes` to `boxes`.

    Args:
        boxes (torch.Tensor): Basic boxes.
        pred_bboxes (torch.Tensor): Encoded boxes with shape
        max_shape (tuple[int], optional): Maximum shape of boxes.
            Defaults to None.
        wh_ratio_clip (float, optional): The allowed ratio between
            width and height.

    Returns:
        torch.Tensor: Decoded boxes.
    """
 
    assert pred_bboxes.size(0) == anchors.size(0)
    assert anchors.size(-1) == pred_bboxes.size(-1) == 5

    dx = pred_bboxes[:, 0]     # 从0开始，每隔5个取一个，取到最后，即取索引0、5、10......
    dy = pred_bboxes[:, 1]
    dw = pred_bboxes[:, 2]
    dh = pred_bboxes[:, 3]
    # 角度的offset
    dangle = pred_bboxes[:, 4]

    # 最大的长宽比,np.log是以e为底的对数函数
    # wh_ratio_clip=16/1000，max_ratio为4.14
    # wh_ratio_clip=1e-6，wh_ratio_clip为13.2
    max_ratio = np.abs(np.log(wh_ratio_clip))

    # ##  查看截断的预测框和对应的anchors
    # inds_dw = (dw <= -max_ratio) | (dw>=max_ratio)
    # inds_dh = (dh <= -max_ratio) | (dh>=max_ratio)
    # inds = inds_dw | inds_dh
    # if torch.any(inds):
    #     print("截断")
    #     print("anchors:", rois[inds])
    #     print("截断前deltas:", deltas[inds])
    
    # clamp，进行截断
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)

    roi_x = anchors[:, 0]
    roi_y = anchors[:, 1]
    roi_w = anchors[:, 2]
    roi_h = anchors[:, 3]
    # anchor的角度
    roi_angle = anchors[:, 4]
    
    # # 经过验证发现，输入参数rois和deltas所在的GPU就不一样，因此导致bug
    # print(f"rois:{rois.device}, deltas:{deltas.device}, dx:{dx.device}, dy:{dy.device}, roi_w:{roi_w.device}, roi_h:{roi_h.device}, roi_angle:{roi_angle.device}")
    # 这种x、y的编解码方法，从来没见过
    if is_encode_relative:
        cosa = torch.cos(roi_angle)
        sina = torch.sin(roi_angle)
        gx = dx * roi_w * cosa - dy * roi_h * sina + roi_x
        gy = dx * roi_w * sina + dy * roi_h * cosa + roi_y
    else:
        gx = dx * roi_w  + roi_x
        gy = dy * roi_h  + roi_y
    
    # 边界框宽高的编解码方法，与YOLOv3一样
    gw = roi_w * dw.exp()
    gh = roi_h * dh.exp()

    # 先获得解码后的角度，然后进行标准化，确保在[-0.25pi, 0.75pi)之间
    ga = np.pi * dangle + roi_angle
    ga = norm_angle(ga)

    decoded_rboxes = torch.stack([gx, gy, gw, gh, ga], dim=-1).view_as(pred_bboxes)

    # if torch.any(inds):
    #     print("解码后:", bboxes[inds])
    
    return decoded_rboxes


def rboxes_encode_stride(anchors, gt_rboxes, stride, is_encode_relative=True):
    '''Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            anchors (torch.Tensor): [N,5], x,y,w,h,theta, 单位为像素值, theta为弧度
            gt_bboxes (torch.Tensor): [N,5], x,y,w,h,theta, 单位为像素值, theta为弧度

        Returns:
            encoded_rboxes(torch.Tensor): Box transformation deltas
        
    '''
    # 保证anchor的个数个gt的个数一样
    assert anchors.size(0) == gt_rboxes.size(0)
    assert anchors.size(-1) == gt_rboxes.size(-1) == 5

    # 下面进行编码
    gt_w = gt_rboxes[..., 2]
    gt_h = gt_rboxes[..., 3]
    gt_angle = gt_rboxes[..., 4]

    anchors_w = anchors[..., 2]
    anchors_h = anchors[..., 3]
    anchors_angles = anchors[..., 4]

    # gt与anchor的中心点坐标的offsets
    xy_offsets = gt_rboxes[..., 0:2] - anchors[..., 0:2]
    ## 计算offsets
    dx = xy_offsets[..., 0] / stride
    dy = xy_offsets[..., 1] / stride
    # torch.log是以e为底的，标准化，但是无法确保归一化
    dw = torch.log(gt_w / anchors_w)
    dh = torch.log(gt_h / anchors_h)
    
    # 角度差转化到[-0.25pi,0.75pi]之间，然后进行归一化
    da = (gt_angle - anchors_angles)
    da = norm_angle(da) / np.pi

    encoded_rboxes = torch.stack((dx, dy, dw, dh, da), -1).reshape(-1,5)

    # means = deltas.new_tensor(means).unsqueeze(0)
    # stds = deltas.new_tensor(stds).unsqueeze(0)
    # deltas = deltas.sub_(means).div_(stds)

    return encoded_rboxes


def rboxes_decode_stride(anchors,
               pred_bboxes,
               stride,
               is_encode_relative=True,
               wh_ratio_clip=16 / 1000 ):
    """Apply transformation `pred_bboxes` to `boxes`.

    Args:
        boxes (torch.Tensor): Basic boxes.
        pred_bboxes (torch.Tensor): Encoded boxes with shape
        max_shape (tuple[int], optional): Maximum shape of boxes.
            Defaults to None.
        wh_ratio_clip (float, optional): The allowed ratio between
            width and height.

    Returns:
        torch.Tensor: Decoded boxes.
    """
 
    assert pred_bboxes.size(0) == anchors.size(0)
    assert anchors.size(-1) == pred_bboxes.size(-1) == 5

    dx = pred_bboxes[:, 0]     # 从0开始，每隔5个取一个，取到最后，即取索引0、5、10......
    dy = pred_bboxes[:, 1]
    dw = pred_bboxes[:, 2]
    dh = pred_bboxes[:, 3]
    # 角度的offset
    dangle = pred_bboxes[:, 4]

    # 最大的长宽比,np.log是以e为底的对数函数
    # wh_ratio_clip=16/1000，max_ratio为4.14
    # wh_ratio_clip=1e-6，wh_ratio_clip为13.2
    max_ratio = np.abs(np.log(wh_ratio_clip))

    # ##  查看截断的预测框和对应的anchors
    # inds_dw = (dw <= -max_ratio) | (dw>=max_ratio)
    # inds_dh = (dh <= -max_ratio) | (dh>=max_ratio)
    # inds = inds_dw | inds_dh
    # if torch.any(inds):
    #     print("截断")
    #     print("anchors:", rois[inds])
    #     print("截断前deltas:", deltas[inds])
    
    # clamp，进行截断
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)

    roi_x = anchors[:, 0]
    roi_y = anchors[:, 1]
    roi_w = anchors[:, 2]
    roi_h = anchors[:, 3]
    # anchor的角度
    roi_angle = anchors[:, 4]


    # x、y的编解码方法
    gx = dx * stride  + roi_x
    gy = dy * stride  + roi_y
    
    # 边界框宽高的编解码方法，与YOLOv3一样
    gw = roi_w * dw.exp()
    gh = roi_h * dh.exp()

    # 先获得解码后的角度，然后进行标准化，确保在[-0.25pi, 0.75pi)之间
    ga = np.pi * dangle + roi_angle
    ga = norm_angle(ga)

    decoded_rboxes = torch.stack([gx, gy, gw, gh, ga], dim=-1).view_as(pred_bboxes)

    # if torch.any(inds):
    #     print("解码后:", bboxes[inds])
    
    return decoded_rboxes
