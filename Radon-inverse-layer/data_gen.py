import os
import cv2
from cv2 import imwrite
from skimage.io import imread
from skimage.transform import radon,iradon,resize
dir_name = '../data/deep_lesion/raw/'
has_processed = 0
import numpy as np
def generate(name):
    global has_processed
    if name.split('.')[-1]!='png':
        return
    matrix = imread(name,True)
    # mask = np.full((512,512),32678,dtype=np.uint16)
    # matrix = matrix-mask
    size = matrix.shape
    matrix = cv2.resize(matrix,(int(size[0]/2),int(size[1]/2)))
    matrix = matrix.astype(np.int32)
    matrix = matrix-32678
    matrix = np.where(matrix>=-1000,matrix,-1000)
    
    #matrix.resize((256,256))
    imwrite("./datas/s%04d.png"%has_processed, matrix)
    #img = imread("./datas/s%04d.png"%has_processed,True)
    #theta = np.linspace(0.,180,256,endpoint=False)
    #radon_img1 = radon(img,theta,circle=True)
    #radon_img = 512*(radon_img1/(2*512*512))
    #imwrite("./datas/r%04d.png"%has_processed,radon_img)
    #ra_img = imread(str(has_processed)+"ra.png",True)
    #recover = iradon(radon_img1,theta,circle=True)
    #cv2.imwrite(str(has_processed)+"reco.png",recover)

    has_processed+=1

def process(name):
    dirs = []
    if os.path.isdir(name):
        dirs = os.listdir(name)
        for dir in dirs:    
            if has_processed>20000:
                return
            if os.path.isdir(name+dir+'/'):
                process(name+dir+'/')
            else:
                generate(name+dir)
    else:
        if has_processed>20000:
            return
        generate(name)

process(dir_name)

