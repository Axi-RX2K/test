
import numpy as np
# from core import Model
from core.model import Model
import cv2
from utils.mask_process import *

YOLO_DETETCT = 1

model = Model(model_name='foundation', 
              camera_config='model_data/configs/camera_config_dechuang_640x480.yaml',
              onnx=True, trt_flag=True,
            #   corr_implementation = 'reg'
              )

def click_event(event, x, y, flags, param):
    x = x % res['threeD'].shape[1]
    y = y % res['threeD'].shape[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at pixel location: ({res['threeD'][y, x]})")

if YOLO_DETETCT:
    from ultralytics import YOLO
    yolo = YOLO('model_data/yolo/yolov8n-seg.pt')
    # yolo = YOLO('model_data/yolo/yolov8-seg-face.pt')
    # yolo = YOLO('model_data/yolo/yolov8n-seg-fire.pt')

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps, t1 = 0, 0
    
    while True:
        # 读取图像
        ret, frame = cap.read()
    
        if not ret:
            print('相机未找到，停止')
            break

        splity = frame.shape[1] // 2
        frame1, frame2 = frame[:, :splity], frame[:, splity:]
        
        # 畸变矫正
        # imgL, imgR = model.remap(frame1, frame2)
        res = model.update(frame1, frame2, rate=1, remap=True)
        imgL, imgR = res['image1'], res['image2']

        if YOLO_DETETCT:
            results = yolo.predict(imgL, conf=0.4)[0]
            decodes = get_decodes(results, yolo.names, res['threeD'], model.camera_config.Q)
            imgL_board  = draw_image(imgL, decodes, mode='whxyz') 
        else:
            imgL_board = imgL


        cv2.putText(imgL_board, 'left', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(imgR, 'right', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        img = np.hstack((imgL_board, res['disp_map']))

        img_st = imgL // 2 + imgR // 2
        cv2.imshow('img_st', img_st)
        cv2.imshow('img', img)
        cv2.setMouseCallback('img', click_event)
        cv2.imshow('imgR', imgR)
        # cv2.imshow('disp_map', imgL
        # cv2.setMouseCallback('imgL', click_event)
        # cv2.imshow('disp', model.disp_map)

        key = cv2.waitKey(1)
        if key == 27: # esc
            break

        if key == 32: # space
            model.show_pcd(res)


    cap.release()
    cv2.destroyAllWindows()