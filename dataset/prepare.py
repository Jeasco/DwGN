import os
import cv2

weathers = ['snow','drop','haze','rain','night']

with open("./train/mix.txt", "w") as a:

    for w in weathers:
        with open(w + "/train/train.txt","w") as f:

            path_name = os.listdir(w + "/train/input")
            for path in path_name:
                f.write('/train/input/' + path + ' ' + '/train/label/' + path + '\n')
                a.write('/' + w + '/train/input/' + path + ' /' + w + '/train/label/' + path + '\n')


        with open(w + "/test/test.txt","w") as f:

            path_name = os.listdir(w + "/test/input")
            for path in path_name:
                f.write('/test/input/' + path + ' ' + '/test/label/' + path + '\n')

