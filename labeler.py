import glob
import cv2
import os
import datetime


paths = glob.glob('./imgs/*.jpg')
timestamp = datetime.datetime.now().strftime('_%y_%m_%d_%H_%M_%S')
for path in paths:
    im = cv2.imread(path) / 255.0
    cv2.imshow('label me', im)
    label = cv2.waitKey(0)
    base = None
    if label == ord('w'):
        base = './imgs/up/'
    elif label == ord('s'):
        base = './imgs/down/'
    elif label == ord('q'):
        break
    else:
        base = './imgs/'
    os.rename(path, base + path.split('/')[-1][:-4]+timestamp+'.jpg')