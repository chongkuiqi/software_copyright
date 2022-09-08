import torch

from pathlib import Path

import cv2
import torch
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    # 获得原始图像的[h,w]
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 缩放比例，新的shape/原始shape, 如果r>1,说明要进行向上缩放
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # 即只缩小，不放大，提高mAP,但是默认情况下是进行向上缩放的
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    # 宽、高同比例缩放
    ratio = r, r  # width, height ratios
    # 同比例缩放后的shape，还没有padding的shape
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # dw,dh是要padding的值
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    # 计算最小矩形框，np.mod()是取余函数，也就是说，虽然进行了padding，但padding后的shape也不一定是设置的new_shape
    # 而是在保证是32的倍数的情况下，取最小的方框，这样padding的像素数最少
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # 分成2半，在两边填充
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # 进行同比例缩放
    # 如果原始图像的shape不是缩放后padding前的shape，进行resize，由此获得同比例缩放后的图像
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # 进行padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    # 返回同比例缩放和padding后的图像、缩放比例、填充值
    return img, ratio, (dw, dh)


img_pathname = "/home/lab/ckq/LARVehicle/LAR1024/train/images/vedai_00001205__1.0__0___0.png"
weights = './model_dep.pt'
device = torch.device("cpu")
img_size = 1024

img0 = cv2.imread(img_pathname)  # BGR
# 对图像进行padding，注意，在测试时，图像进行同比例缩放后，还会进行padding，
# 但不一定padding到self.img_size大小，而是保证填充的像素数最少
# Padded resize
img = letterbox(img0, new_shape=img_size)[0]
# Convert
# BGR变为RGB，shape变为[channel,h,w]
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
# ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
img = np.ascontiguousarray(img)

# 图像数据转化为tensor，并放入设备中
img = torch.from_numpy(img).to(device)
img = img.float()  # uint8 to fp16/32
img /= 255.0  # 0 - 255 to 0.0 - 1.0
if img.ndimension() == 3:
    # 增加一个维度，其实就是batch维度
    img = img.unsqueeze(0)



model_pathname = "./runs/train/exp248/weights/best.pt"
model = torch.load(model_pathname, map_location=device)['model'].float().eval()
pred = model(img)

# print(pred)

'''
    使用torch.jit.trace进行模型保存和加载，使得模型的推理不依赖模型结构源码
'''
# trace_module = torch.jit.trace(model, img) 
# print(trace_module.code)  # 查看模型结构
# output = trace_module(img) # 测试
# # print(output)

# #  保存jit格式的模型
# trace_module.save("model_jit.pt")

# # 加载jit格式的模型
# model = torch.jit.load("model_jit.pth", map_location=device)
# # 运行报错，显示DeformConvFunction函数错误
# output = model(img)
# # print(output)


'''
    使用torch.jit.script方法进行模型保存和加载，使得模型的推理不依赖模型结构源码
'''
script_module = torch.jit.script(model) 
print(script_module.code)
output = script_module(img)
print(output)

# trace_modult('model.pt') # 模型保存

# # 此时应该用script方法  模型定义有 if else  等控制语句
# script_module = torch.jit.script(model) 
# print(script_module.code)
# output = script_module(torch.rand(1,1,224,224))
# print(output)
# script_modult('model.pt') # 模型保存

