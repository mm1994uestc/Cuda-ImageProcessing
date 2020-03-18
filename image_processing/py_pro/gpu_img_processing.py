from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2 as cv

mod = SourceModule("""
        __global__ void ImgSub(float *A, float *B, int w, int h, int d)
        {
          int idx = blockIdx.x*blockDim.x + threadIdx.x;
          int idy = blockIdx.y*blockDim.y + threadIdx.y;
          float tmp = 0;
          int line,strip;
          line = idy*w*d;
          strip = idx*d;
          if(idx < w && idy < h){
            tmp += *(A+line+strip) - *(B+line+strip);
            tmp += *(A+line+strip+1) - *(B+line+strip+1);
            tmp += *(A+line+strip+2) - *(B+line+strip+2);
            if(tmp > 75 || tmp < -75){
              *(A+line+strip) = 0;
              *(A+line+strip+1) = 0;
              *(A+line+strip+2) = 0;
            }
          }
        }
        """)

cv.namedWindow("Moving Detecting!")
cap = cv.VideoCapture(0)

W = 320
H = 240
D = 3
cap.set(cv.CAP_PROP_FRAME_HEIGHT,H)
cap.set(cv.CAP_PROP_FRAME_WIDTH,W)

ret, frame = cap.read()
f_old = frame.astype(np.float32)
f_old = f_old.ravel('C')
f_old = f_old.astype(np.float32)
f_bytes = W*H*D*4

w = np.int32(W)
h = np.int32(H)
d = np.int32(D)

blockDimX = 16
blockDimY = 16
gridDimX = D * W / blockDimX
gridDimY = H / blockDimY

f_old_gpu = cuda.mem_alloc(f_bytes)
f_new_gpu = cuda.mem_alloc(f_bytes)
cuda.memcpy_htod(f_old_gpu, f_old) # dest src
f_old_gpu = gpuarray.to_gpu(f_old)
Moving_Detect = mod.get_function("ImgSub")

while True:
    ret, frame = cap.read()
    f_new = frame.astype(np.float32)
    f_new = f_new.ravel('C')
    cuda.memcpy_htod(f_new_gpu, f_new)
    Moving_Detect(f_new_gpu, f_old_gpu, w, h, d, block=(blockDimX,blockDimY,2), grid=(gridDimX,gridDimY))
    f_old_gpu = gpuarray.to_gpu(f_new)
    cuda.memcpy_dtoh(f_new, f_new_gpu)
    f_new = f_new.reshape([H,W,D])
    frame = f_new.astype(np.uint8)
    cv.imshow("Moving Detecting!",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
