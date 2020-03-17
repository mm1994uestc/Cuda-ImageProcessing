## PyCUDA Installing
The pycuda reference:![PyCuda-Tutorial](https://documen.tician.de/pycuda/tutorial.html)
For Jetson Nano,to install pycuda should do:
1. Firstly setup the cuda-library(Jetson Nano JetPack had already install cuda-lib).
```
export CUBA_HOME=/usr/local/cuda-10.0
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
```
2. Secondly install the python-pip app for Jetson Nano.
`sudo apt-get install python-pip`
3. Thirdly install pycuda by pip.
`pip install pycuda`

import os,math,cv2,numpy
from PIL import Image
import numpy as np
import skimage
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from timeit import default_timer as timer

mod = SourceModule("""
__global__ void Gauss_cuda(int ***img,int ***im,float **template,int template_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;
    int count = 0;
    for(int i=x;i<x+template_size;i++)
    {
        for(int j=y;j<y+template_size;j++)
        {
            count += im[i,j,z]*template[i,j,z]; 
        }
    }
    img[x][y][z] = count;  
}
__global__ void bilateral_cuda(float ***img,float ***orig,float **template,float border,float sigma1,float sigma2,int rows,int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;
    int xmin = max(x-border,0);
    int xmax = min(x+border+1,rows);
    int ymin = max(y-border,0);
    int ymax = min(y+border+1,cols);
    int w = 0;
    int v = 0;
    for(int i=xmin;i<xmax;i++)
    {
        for(int j=ymin;j<ymax;j++)
        {
            w += w+template[xmin-x+border+i,ymin-y+border+i]*exp((orig[i,j,z]-orig[x,y,z])*(orig[i,j,z]-orig[x,y,z])/(2*sigma2*sigma2));
            v += orig[x][y][z]*w;               
        }
    }
    img[x][y][z] = v/w;  
}

""")

Gauss_cuda = mod.get_function("Gauss_cuda")
bilateral_cuda = mod.get_function("bilateral_cuda")


def generateGaussianTemplate(template_size,sigma):
    template = np.zeros((template_size,template_size))
    mid = template_size/2
    sum = 0
    pi = 3.1415926
    for i in range(template_size):
        x = pow(i-mid,2)
        for j in range(template_size):
            y = pow(j-mid,2)
            g = math.exp((x+y)/(-2*sigma*sigma))/(2*pi*sigma)
            template[i][j] = g
            sum+=g
    #归一化
    template = template/sum
    return template

def Gauss_filter(img,template_size,sigma):
    [rows,cols,channel] = img.shape
    border = int(template_size/2)
    template = generateGaussianTemplate(template_size,sigma)
    im = cv2.copyMakeBorder(img,border,border,border,border,cv2.BORDER_REPLICATE)
    Gauss_cuda(
                drv.InOut(img), drv.In(im), drv.In(template),template_size,
                block=(template_size,template_size,3),grid=(rows,cols))
    return img

def Bilateral_filter(img,template_size,sigma1,sigma2):
    img = img/255
    border = int(template_size/2)
    [rows,cols,channel] = img.shape
    tmp = np.arange(-border,border+1)
    [x,y] = np.meshgrid(tmp,tmp)
    X = np.power(x,2)
    Y = np.power(y,2)
    d = np.exp(-(X+Y)/(2*sigma1*sigma1))
    orig_img = img
    Gauss_cuda(
                drv.InOut(img), drv.In(orig_img), drv.In(d),border,sigma1,sigma2,rows,cols,
                block=(template_size,template_size,3),grid=(rows,cols))
    return img*255

if __name__=='__main__':
    img = cv2.imread('demo2.jpg')
    #noise = skimage.util.random_noise(img,'gaussian')*255
    #cv2.imwrite('noise.jpg',noise)
    template_size = 3
    sigma_g = 0.8
    sigma = [3,0.1]
    bilateral = Bilateral_filter(img,template_size,sigma[0],sigma[1])
    Gauss = Gauss_filter(img,template_size,sigma_g)
    cv2.imwrite('bilateral_result_{}_{}_{}.jpg'.format(template_size,sigma[0],sigma[1]),bilateral)
    cv2.imwrite('Gauss_result_{}_{}.jpg'.format(template_size,sigma_g),Gauss)
