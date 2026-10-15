
import numpy as np
from core import Model
import cv2
from utils.mask_process import *

model = Model(model_name='cre', device='cuda', 
              onnx=False,
              camera_config='model_data/configs/camera_config_jmy_720p.yaml')

def click_event(event, x, y, flags, param):
    x = x % model.threeD.shape[1]
    y = y % model.threeD.shape[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at pixel location: ({model.disparity[y, x]})")

YOLO_DETETCT = 0
if YOLO_DETETCT:
    from ultralytics import YOLO
    yolo = YOLO('model_data/yolo/yolo11n-seg.pt')

frame1 = cv2.imread('testfile/zed_left_1748247842765.png')
frame2 = cv2.imread('testfile/zed_right_1748247842765.png')

# frame  = cv2.imread(r'testfile\shts\6c83f4e6e0dd1b69124c6eb70d0cb44.jpg')
# splity = frame.shape[1] // 2
# frame1, frame2 = frame[:, :splity], frame[:, splity:]

res = model.update(frame1, frame2, rate=0.92, remap=True)

imgL, imgR = res['image1'], res['image2']



if YOLO_DETETCT:
    results = yolo.predict(imgL, conf=0.4)[0]
    decodes = get_decodes(results, yolo.names, res['threeD'], model.camera_config.Q)
    imgL_board  = draw_image(imgL, decodes) 
else:
    imgL_board = imgL


cv2.putText(imgL_board, 'left', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(imgR, 'right', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

img_st = imgL // 2 + imgR // 2

# print(np.mean(res['disp_map']))
img = np.hstack((imgL_board, res['disp_map']))
cv2.imshow('img', img)
cv2.imshow('img_st', img_st)
# cv2.setMouseCallback('img', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

model.show_pcd(res['points'], res['framecolor'])
# print(frame1.shape)