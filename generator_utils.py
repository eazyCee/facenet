import os
import cv2
import numpy as np
from parameters import *
import pickle


input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
with open("./path_dict.p", 'rb') as f:
    paths = pickle.load(f)
    
faces = []
for key in paths.keys():
    paths[key] = paths[key].replace("\\", "/")
    faces.append(key)
    
images = {}
for key in paths.keys():
    li = []
    for img in os.listdir(paths[key]):
        img1 = cv2.imread(os.path.join(paths[key],img))
        if img1 is not None:
            img2 = img1[...,::-1]
            li.append(np.around(np.transpose(img2, (2,0,1))/255.0, decimals=12))
        else:
            print(f"Error loading image: {img}")
    images[key] = np.array(li)


def batch_generator(batch_size=16):
    y_val = np.zeros((batch_size, 2, 1))
    anchors = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    positives = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    negatives = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    
    while True:
        for i in range(batch_size):
            positiveFace = faces[np.random.randint(1,len(faces))]
            negativeFace = faces[np.random.randint(1,len(faces))]
            while positiveFace == negativeFace:
                negativeFace = faces[np.random.randint(1,len(faces))]
            
            
            # positives[i] = images[positiveFace][np.random.randint(np.random.randint(1,3), None)]
            # anchors[i] = images[positiveFace][np.random.randint(np.random.randint(1,3), None)]
            # negatives[i] = images[negativeFace][np.random.randint(np.random.randint(1,3), None)]
            
            positives[i] = images[positiveFace][np.random.randint(len(images[positiveFace]), None)]
            anchors[i] = images[positiveFace][np.random.randint(len(images[positiveFace]), None)]
            negatives[i] = images[negativeFace][np.random.randint(len(images[negativeFace]), None)]
        
        x_data = {'anchor': anchors,
                  'anchorPositive': positives,
                  'anchorNegative': negatives
                  }
        
        yield (x_data, [y_val, y_val, y_val])