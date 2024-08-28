from __future__ import print_function
from facenet_pytorch import MTCNN, InceptionResnetV1,extract_face
from PIL import Image,ImageDraw
import torch
import cv2
import os
import torch.nn as nn
import utils
import numpy as np
from scipy.spatial import distance as dis
import pickle
import argparse
import time


with open('./models/EVM_model.pkl','rb') as b:
    mevm = pickle.load(b)

with open('./models/class_names.pkl','rb') as cl:
    cls_names = pickle.load(cl)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
mtcnn = MTCNN(
    image_size=160, margin=20, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    device=device, keep_all = True,
)

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)



def verify2(embedding,frame,boxes):
    for j,l in enumerate(embedding):
      # Computing Cosine distance.
        l = l.reshape(1,-1)
        probs,index = mevm.max_probabilities(l)
      # Chosen threshold is 0.85. 
      #Threshold is determined after seeing the table in the previous cell.
        print(probs,cls_names[index[0][0]]) 
      # names = out_encoder.inverse_transform([cls_names[index[0][0]]])
      # print(names)
        if probs[0] > 0.8:
            text = cls_names[index[0][0]]
          #Name of the person identified is printed on the screen, as well as below the detecetd face (below the rectangular box).
            cv2.putText(frame, text,(int(boxes[j][0]) ,int(boxes[j][3]) + 17), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
        else:
            text = 'unknown'
            cv2.putText(frame, text,(int(boxes[j][0]) ,int(boxes[j][3]) + 17), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)

               
                

    
def capture(image_path):                    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    frame = cv2.imread(image_path)

    frames = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    img_cropped = mtcnn(frames)
    print(img_cropped.shape)
    boxes,prob= mtcnn.detect(frames)
    frame0 = utils.draw_bbox(boxes, frame)
    img_cropped = img_cropped.unsqueeze(0)
    for img_embedding in img_cropped:
        img_embeddings = resnet(img_embedding.to(device))
        # print(img_embedding.shape)
        verify2(img_embeddings.detach().cpu(),frame0,boxes)

    print(time.time())

    cv2.imwrite('Detect.jpg',frame0)

    key = cv2.waitKey(1)
    if key ==13: 
        cv2.destroyAllWindows()


if __name__ == '__main__':

      parser = argparse.ArgumentParser()
      parser.add_argument('--image_path',default=None,required=True,
                          help = "Enter the path of test images")
      args = parser.parse_args()

      capture(args.image_path)

