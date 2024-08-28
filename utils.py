import cv2
import pickle


# draw the bounding boxes for face detection
def draw_bbox(bounding_boxes, image):
    for i in range(len(bounding_boxes)):
        x1, y1, x2, y2 = bounding_boxes[i]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    return image


def data_load():
    with open("./models/EVM_model.pkl", "rb") as b:
        mevm = pickle.load(b)

    with open("./models/class_names.pkl", "rb") as cl:
        cls_names = pickle.load(cl)

    return (mevm, cls_names)


def calc_dist(embeddings, mevm, cls_names, frame, x, y):
    for j, l in enumerate(embeddings):
        # Computing Cosine distance.
        l = l.reshape(1, -1)
        probs, index = mevm.max_probabilities(l)
        # Chosen threshold is 0.85.
        # Threshold is determined after seeing the table in the previous cell.
        print(probs, cls_names[index[0][0]])
        # names = out_encoder.inverse_transform([cls_names[index[0][0]]])
        # print(names)
        if probs[0] > 0.8:
            text = cls_names[index[0][0]]
            # Name of the person identified is printed on the screen, as well as below the detecetd face (below the rectangular box).
            cv2.putText(
                frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2
            )
        else:
            text = "unknown"
            cv2.putText(
                frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2
            )

    return (text, frame)
