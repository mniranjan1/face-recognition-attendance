from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import cv2
import numpy as np
import pickle
from statistics import mode

with open("./models/EVM_model.pkl", "rb") as b:
    mevm = pickle.load(b)

with open("./models/class_names.pkl", "rb") as cl:
    cls_names = pickle.load(cl)


# Model running on CPU
device = torch.device("cpu")

# Define Inception Resnet V1 module (GoogLe Net)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    select_largest=True,
    device=device,
)

n_conform = 5
iteration = []

f_conformation = []

text_switch = False


def get_face(img):
    modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "models/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    image = img
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
    )
    net.setInput(blob)
    faces = net.forward()

    # to draw faces on image
    temp_img = []
    boxes = []
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.9:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            temp_img.append(image[y:y1, x:x1])
            boxes.append((x, y, x1, y1))

    prod = 1
    t_i_index = -1
    for i, each_img in enumerate(temp_img):
        m, n = each_img.shape[:2]
        if prod < (m * n):
            prod = m * n
            t_i_index = i
    return cv2.resize(temp_img[t_i_index], (160, 160)), boxes[t_i_index]


def verify2(embedding, frame):
    for _, main_embedding in enumerate(embedding):
        # Computing Cosine distance.
        main_embedding = main_embedding.reshape(1, -1)
        probs, index = mevm.max_probabilities(main_embedding)
        # Chosen threshold is 0.85.
        # Threshold is determined after seeing the table in the previous cell.
        print(probs, cls_names[index[0][0]])
        # names = out_encoder.inverse_transform([cls_names[index[0][0]]])
        # print(names)
        if probs[0] > 0.8:
            text = cls_names[index[0][0]]
            # Name of the person identified is printed on the screen,
            # as well as below the detecetd face (below the rectangular box).
            cv2.putText(
                frame,
                text,
                (boxes[0], boxes[1]),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (255, 255, 255),
                2,
            )
        else:
            text = "unknown"
            cv2.putText(
                frame,
                text,
                (boxes[0], boxes[1]),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (255, 255, 255),
                2,
            )

        return text


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, image = cap.read()
    image = cv2.flip(image, 1)
    if ret:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames = Image.fromarray(rgb)
        switch = True
        try:
            _, boxes = get_face(rgb)
            cv2.rectangle(
                image, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (100, 100, 100), 3
            )
            img_cropped = mtcnn(frames)
            img_cropped = img_cropped.unsqueeze(0)
            img_embedding = resnet(img_cropped)
            label = verify2(img_embedding.detach().numpy(), image)
            iteration.append(label)
            if len(iteration) == n_conform:
                c_label = mode(iteration)
                iteration = []
                text_switch = True
                f_conformation.append(c_label)
                # print(f_conformation)
            switch = False
        except Exception as e:
            print(e)
            text_switch = False
            iteration = []
            f_conformation = []

        if switch:
            image = image
        cv2.imshow("EyeTrack", image)

        try:
            if len(f_conformation) == 5:
                print(mode(f_conformation))
                f_conformation = []
        except Exception as e:
            print(e)

    key = cv2.waitKey(1)
    # 13 is for 'Enter' key.
    # If 'Enter' key is pressed, all the windows are made to close forcefully.
    if key == 13:
        break
cap.release()
cv2.destroyAllWindows()
