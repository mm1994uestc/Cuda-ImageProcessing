import pycuda.gpuarray as gpuarray
import pycuda.cumath as cmath
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2 as cv

cv.namedWindow("Moving Detecting!")
cap = cv.VideoCapture(0)

W = 320
H = 240

cap.set(cv.CAP_PROP_FRAME_HEIGHT,H)
cap.set(cv.CAP_PROP_FRAME_WIDTH,W)

ret, frame = cap.read()
gray_a = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)

img_ori_gpu = gpuarray.to_gpu(gray_a.astype(np.float32))
img_buf_gpu = gpuarray.empty_like(img_ori_gpu)
img_sub = gpuarray.ones_like(img_ori_gpu)
img_sub = 25*img_sub
img_bgm = gpuarray.zeros_like(img_sub)
while True:
    ret, frame = cap.read()
    gray_buff = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    img_res_gpu = gpuarray.to_gpu(gray_buff.astype(np.float32))
    img_buf_gpu = cmath.fabs(img_ori_gpu - img_res_gpu)
    img_buf_gpu = img_buf_gpu - img_sub
    img_ori_gpu = img_res_gpu.copy()
    img_res_gpu = gpuarray.if_positive(img_buf_gpu,img_bgm,img_res_gpu)
    gray_buff = img_res_gpu.get()
    gray_buff = gray_buff.astype(np.uint8)
    frame = cv.cvtColor(gray_buff,cv.COLOR_GRAY2RGB)
    cv.imshow("Moving Detecting!",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
