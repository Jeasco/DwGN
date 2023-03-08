import os
import cv2

with open("./ImageNet/imagenet.txt","w") as f:
    for i in range(1000):
        path_name = os.listdir("./ImageNet/" + str(i))
        for path in path_name:
            f.write("/" + str(i) +path+'\n')


