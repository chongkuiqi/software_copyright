'''
(1) designer 路径: /home/lab/anaconda3/lib/python3.7/site-packages/qt5_applications/Qt/bin/designer
(2) .ui文件转化为.py文件：pyuic5 -o MainGUI.py MainGUI.ui
'''

# 导入程序运行必须模块
import sys
# from time import sleep
# PyQt5中使用的基本控件都在PyQt5.QtWidgets模块中
from PyQt5.QtWidgets import QApplication, QMainWindow
# from cv2 import namedWindow

#导入designer工具生成的模块，这个模块只定义了主窗口以及控件，并没有程序入口的代码，以实现UI界面与业务逻辑分离的功能
from MainGUI import Ui_MainWindow


import numpy as np
import cv2
import torch
import os
import math

def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
                     + cal_line_length(combinate[i][1], dst_coordinate[1]) \
                     + cal_line_length(combinate[i][2], dst_coordinate[2]) \
                     + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return np.array(combinate[force_flag]).reshape(8)


def rotated_box_to_poly_single(rrect):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x_ctr, y_ctr, width, height, angle = rrect[:5]
    tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
    rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    poly = R.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
    poly = get_best_begin_point_single(poly)
    return poly


def rotated_box_to_poly_np(rrects):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    polys = []
    for rrect in rrects:
        poly = rotated_box_to_poly_single(rrect)
        polys.append(poly)
    
    polys = np.array(polys).reshape(-1,8)
    return polys


def plot_rotate_boxes(img, boxes_points, cls_fall_point=0, color=(0, 0, 255), thickness=1):
    '''
        画旋转框，以及目标类别
    inputs:
        img:
        box_points: shape:[N,9] or [N,8], np.float64类型
        classes_name:
        cls_fall_point:表示，目标类别文本画在旋转框四个角点的哪个角上
    return:
        img
    '''
    num_cols = boxes_points.shape[1]
    # 逐个画box
    for box_points in boxes_points:
        if num_cols == 9:
            # 目标类别
            cls_id = int(box_points[0])
            box_points = box_points[1:].reshape(4,2).astype(np.int32)
        elif num_cols == 8:
            box_points = box_points.reshape(4,2).astype(np.int32)
        # 画4个点组成的轮廓，也就是旋转框
        # cv2.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
        # contours:轮廓列表，每个元素都是一个轮廓, 元素可以是一个numpy数组，
        # contourIdx：表示画列表中的哪个轮廓，-1表示画所有轮廓
        # thickness：如果是-1（cv2.FILLED），则为填充模式。
        cv2.drawContours(img, [box_points], contourIdx=-1, color=color, thickness=thickness)
        
        # if cls_fall_point <0 or cls_fall_point > 3:
        #     print(f"cls_fall_pont must be 0-3 !!!")
        #     exit()
        # pt1 = tuple(box_points[cls_fall_point])
        # # cv2.putText，按照顺序，各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        # cv2.putText(img, 
        #     text = classes_name[cls_id], 
        #     org = pt1, 
        #     fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
        #     fontScale = 1, 
        #     color = color, 
        #     thickness = thickness
        # )

    return img


# 将输入图像尺寸上的旋转框坐标，转化为原始图像尺寸上的坐标
def scale_coords_rotated(img1_shape, bboxes, img0_shape, ratio_pad=None):
    
    # Rescale bboxes from img1_shape to img0_shape
    '''
    bboxes : torch.tensor, [N,6(x,y,w,h,theta,score)]
    '''
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    bboxes[:, 0] -= pad[0]  # x padding
    bboxes[:, 1] -= pad[1]  # y padding

    bboxes[:, :4] /= gain

    # 不再进行坐标截断
    # clip_coords(bboxes, img0_shape)
    return bboxes


def img_batch_normalize(img):
    # img_normalize_mean =  np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3,1,1)
    # img_normalize_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3,1,1)
    # img = (img - img_normalize_mean) / img_normalize_std

    img_normalize_mean =  [123.675, 116.28, 103.53]
    img_normalize_std = [58.395, 57.12, 57.375]
    
    img[:,0,:,:] = (img[:,0,:,:] - img_normalize_mean[0]) / img_normalize_std[0]
    img[:,1,:,:] = (img[:,1,:,:] - img_normalize_mean[1]) / img_normalize_std[1]
    img[:,2,:,:] = (img[:,2,:,:] - img_normalize_mean[2]) / img_normalize_std[2]

    return img


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


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(im, img_size):
   
    h0, w0 = im.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                        interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized


# 用于显示UI主窗口
class ShowUIMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(ShowUIMainWindow, self).__init__(parent)
        self.setupUi(self)

        # 添加 "开始检测"按钮的信号和槽
        # 将"开始检测"按钮的点击操作，与self.detect_img程序的运行，连接起来
        # start_detect_button表示设计UI界面的中的"开始检测"按钮的设置的对象名字
        # clicked表示鼠标左键点击的信号
        self.start_detect_button.clicked.connect(self.detect_img)

        # 添加 "退出"按钮的信号和槽
        self.exit_button.clicked.connect(self.exit)

        # 提示用户输入图像路径
        self.dispaly_log.setText("请输入要检测的图像路径")
        
    
    def exit(self):
        cv2.destroyAllWindows()
        sys.exit()


    # 调用模型，并获得检测结果
    def detect_img(self):

        # append方法不会覆盖之前的文本
        # self.dispaly_log.setText(f"获取待检测图像的路径......")
        self.dispaly_log.append(f"获取待检测图像的路径......")
        # 获得输入的图像路径
        # "vedai_00000373__1.0__0___0.png"
        img_pathname = self.img_pathname.text()

        # 
        self.img = cv2.imread(img_pathname)
        self.visual_img(self.img.copy())

        img, shape0, ratio_pad = self.preprocess_img()
        self.model = self.get_model()

        device = torch.device("cuda:0")
        half = True
        if device is not None:
            img = img.to(device, non_blocking=True)
            self.model.to(device)

            if half:
                img = img.half()
                self.model.half()

        self.dispaly_log.append(f"正在进行检测......")
        img_results = self.model(img, post_process=True)

        # 后处理
        # det_bboxes shape : [N, 6(x,y,w,h,theta,score)], labels shape:[N,1]
        det_bboxes, det_labels = img_results[0]
        # 输入图像尺寸上的边界框坐标，转化为原始图像尺寸上的坐标
        det_bboxes = scale_coords_rotated(img[0].shape[1:], det_bboxes, shape0, ratio_pad)
        det_bboxes = det_bboxes.cpu().numpy()
        det_bboxes_points = rotated_box_to_poly_np(det_bboxes[:,:5]).reshape(-1,8)
        det_scores = det_bboxes[:,-1].reshape(-1,1)
        det_labels = det_labels.cpu().numpy().reshape(-1,1)
        
        self.dispaly_log.append(f"{img_pathname}检测完成！")

        # 清空文本
        # self.dispaly_log.clear()

        ## 保存检测结果
        self.save_img_visual(img_pathname, det_bboxes_points)
        self.save_txt_result(img_pathname, det_bboxes_points, det_scores, det_labels)

    # 获取图像，包括图像的读取、图像的shape的更改，以及转化为torch.tensor数据，并进行图像的归一化
    def preprocess_img(self, img_size=1024):
        

        img, (h0, w0), (h, w) = load_image(self.img.copy(), img_size)

        img, ratio, pad = letterbox(img, img_size, auto=False, scaleup=False)
        shape0 = (h0, w0)
        ratio_pad = ((h / h0, w / w0), pad)  
        
        # Convert
        # BGR变为RGB，shape变为[channel,h,w]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
        img = np.ascontiguousarray(img)

        # 图像数据转化为tensor，并放入设备中
        img = torch.from_numpy(img)
        img = img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            # 增加一个维度，其实就是batch维度
            img = img.unsqueeze(0)
        
        img = img_batch_normalize(img)

        return img, shape0, ratio_pad


    def get_model(self, weight_path="/home/lab/ckq/S2ANet/runs/train/exp252/weights/best.pt"):

        model = torch.load(weight_path, map_location='cpu')['model']
        model.float()

        # 设置检测算法的参数
        model.head.score_thres_before_nms = float(self.score_thr.text())
        model.head.iou_thres_nms = float(self.iou_thr.text())
        model.head.max_per_img = int(self.max_per_img.text())

        return model


    def visual_img(self, img):
        window_name = "image"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)


    def save_img_visual(self, img_pathname, boxes_points):
        
        img = self.img.copy()
        img = plot_rotate_boxes(img, boxes_points, thickness=3)

        self.visual_img(img)

        name, ext = os.path.splitext(img_pathname)
        save_img_pathname = name + '_result' + ext
        cv2.imwrite(save_img_pathname, img)
        return img


    def save_txt_result(self, img_pathname, det_bboxes_points, det_scores, det_labels):
        
        name, ext = os.path.splitext(img_pathname)
        save_txt_pathname = name + '_result.txt'
        lines = []
        for box_points, box_score, box_label in zip(det_bboxes_points, det_scores, det_labels):
            box_points = list(map(str, box_points.tolist()))
            box_score = str(box_score.item())
            box_label = self.model.names[int(box_label)]

            lines.append(
                ' '.join(box_points + [box_score, box_label+'\n'])
            )
            
        with open(save_txt_pathname, 'w') as f:
            f.writelines(lines)

def main():
    # 固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    
    # 初始化
    mainWin = ShowUIMainWindow()
    # 在屏幕上显式窗口，但并不会显示定义的那些窗口控件
    mainWin.show()

    # app.exec_()程序运行，并显示自定义的窗口控件
    app.exec_()
    # sys.exit方法确保程序完整退出。只用app.exec_()也能保证退出
    # sys.exit(app.exec_())

    
if __name__ == "__main__":
    main()
