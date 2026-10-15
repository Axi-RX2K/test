import numpy as np
from core import Model
from utils.CameraConfig import CameraConfig
# import model_data.cameraconfig_video as camera_config
import cv2
import os
from utils.mask_process import draw_image, get_decodes
import imageio
import platform
import gc

def video_depth(input_video, model_name, yaml_path, detect=1,onnx=False):
    current_os = platform.system()
    print(f'当前操作系统：{current_os}')
    # camera_config = CameraConfig(yaml_path)
    # import model_data.cameraconfig_video as camera_config
    model = Model(model_name=model_name, camera_config=yaml_path, onnx=onnx)
    if detect:
        from ultralytics import YOLO
        yolo = YOLO('./model_data/yolo/yolov8n-seg.pt')

    # dir_path = os.path.dirname(input_video)
    filename, ext = os.path.splitext(input_video)
    output_video_path = os.path.join(f'{filename}-{model_name}.mp4')
    print(output_video_path)
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    # exit()
    
    if current_os == "Linux":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if current_os == "Windows":
        writer = imageio.get_writer(output_video_path, mode='I', fps=fps)

    while True:
        # 读取图像
        ret, frame = cap.read()
        if not ret:
            print('停止')
            break

        splity = frame.shape[1] // 2
        frame1, frame2 = frame[:, :splity], frame[:, splity:]

        # 畸变矫正
        res = model.update(frame1, frame2, rate=1, remap=True)
        imgL, imgR = res['image1'], res['image2']
        # model.get_disparity(imgL, imgR)
        # model.get_depthmap()
        
        if detect:
            results = yolo.predict(imgL, conf=0.6)[0]
            decodes = get_decodes(results, yolo.names, res['threeD'], model.camera_config.Q)
            imgL = draw_image(imgL, decodes, 'm', 2, 'whxyz')

        cv2.putText(imgL, 'left', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(imgR, 'right', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        img = np.hstack((imgL, res['disp_map']))
        
        if current_os == "Windows":
            writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if current_os == "Linux":
            out.write(img)

        cv2.imshow('img', img)
        # cv2.imshow('disp', model.disp_map)

        key = cv2.waitKey(1)
        if key == 27:  # esc
            break

        if key == 32:  # space
            model.show_pcd(res)

    cap.release()
    cv2.destroyAllWindows()
    del model
    gc.collect()
    if current_os == "Linux":
        out.release()
    if current_os == "Windows":
        writer.close()
    # out.release()



if __name__ == '__main__':
    
    # model_names = ['sgbm', 'psm', 'unimatch', 'raft', 'sea-raft', 'cre', 'foundation', 'monster']
    model_names = ['foundation']
    # yaml_path = 'model_data/configs/camera_config_dechuang_640x480.yaml'
    # input_video = 'testfile/videos/20250602133949.mp4'

    yaml_path = 'model_data/configs/camera_config_car.yaml'
    input_video = 'testfile/videos/car.avi'
    for model_name in model_names:
        video_depth(input_video, model_name, yaml_path)

