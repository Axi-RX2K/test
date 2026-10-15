import numpy as np
from core import Model
import cv2
from utils.mask_process import *

model = Model(model_name='cre', onnx=False)
frame1 = cv2.imread('testfile/fq/1.jpg')
frame2 = cv2.imread('testfile/fq/2.jpg')
frame1 = cv2.resize(frame1, (640, 480))
frame2 = cv2.resize(frame2, (640, 480))
imgL, imgR = frame1, frame2
res = model.update(imgL, imgR)
cv2.putText(imgL, 'left', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(imgR, 'right', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
# img = np.hstack((imgL, res['disp_map']))
cv2.imshow('img', imgL)
cv2.imshow('disp', res['disp_map'])
key = cv2.waitKey(0)
cv2.destroyAllWindows()

