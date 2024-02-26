
import torch
from gtts import gTTS
import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import os
from playsound import playsound
# Download the MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# def speak_text(text):
#     tts = gTTS(text=text, lang='en')
#     filename = "voice.mp3"
#     tts.save(filename)
#     playsound(filename)

# Distance constants
# Distance constants
KNOWN_DISTANCE1 = 21 #INCHES
KNOWN_DISTANCE2 = 24 #INCHES
KNOWN_DISTANCE3 = 68 #INCHES

TOOTHBRUSH_WIDTH = 1 #INCHESq
STOPSIGN_WIDTH = 15 #INCHES
REFRIGERATOR_WIDTH = 23


# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3



# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX


# getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)




# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s : %f" % (class_names[int(classid)], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 79:  # person class id
            data_list.append([class_names[int(classid)], box[2], (box[0], box[1] - 2)])
        elif classid == 11:
            data_list.append([class_names[int(classid)], box[2], (box[0], box[1] - 2)])
        elif classid == 72:
            data_list.append([class_names[int(classid)], box[2], (box[0], box[1] - 2)])
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data.
    return data_list



def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = ((real_object_width * focal_length) / width_in_frmae)
    return distance


# reading the reference image from dir
ref_toothbrush = cv.imread('ReferenceImages/image6.png')
ref_stopsign = cv.imread('ReferenceImages/image1.png')
ref_regregerator = cv.imread('ReferenceImages/image11.png')

toothbrush_data = object_detector(ref_toothbrush)
toothbrush_width_in_rf = toothbrush_data[0][1]

stopsign_data = object_detector(ref_stopsign)
stopsign_width_in_rf = stopsign_data[0][1]

regregerator_data = object_detector(ref_regregerator)
regregerator_width_in_rf = regregerator_data[0][1]



print(f"toothbrush width in pixels : {toothbrush_width_in_rf},stopsign width in pixel: {stopsign_width_in_rf},regregerator width in pixel: {regregerator_width_in_rf}")



# finding focal length

focal_toothbrush = focal_length_finder(KNOWN_DISTANCE1, TOOTHBRUSH_WIDTH, toothbrush_width_in_rf)
print(focal_toothbrush)
focal_stopsign = focal_length_finder(KNOWN_DISTANCE2, STOPSIGN_WIDTH, stopsign_width_in_rf)
print(focal_stopsign)
focal_refregerator = focal_length_finder(KNOWN_DISTANCE3, REFRIGERATOR_WIDTH, regregerator_width_in_rf)
print(focal_refregerator)



cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    data = object_detector(frame)
    start = time.time()

    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    input_batch = transform(img).to('cpu')

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic",
                                                     align_corners=False, ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_map = cv.normalize(depth_map, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    depth_map = (depth_map * 255).astype(np.uint8)
    depth_map = cv.applyColorMap(depth_map, cv.COLORMAP_MAGMA)

    for d in data:
        if d[0] == 'toothbrush':

            distance = distance_finder(focal_toothbrush, TOOTHBRUSH_WIDTH, d[1])
            print("toothbrush", d[1])
            x, y = d[2]
        elif d[0] == 'stop sign':
            distance = distance_finder(focal_stopsign, STOPSIGN_WIDTH, d[1])
            x, y = d[2]
            # print(d[1])

        elif d[0] == 'refrigerator':
            distance = distance_finder(focal_refregerator, REFRIGERATOR_WIDTH, d[1])
            x, y = d[2]
            # print(d[1])

        cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
        cv.putText(frame, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, RED, 2)
        # text = "The distance from the " + d[0] + " is " + str(round(distance, 2)) + " inches."
        # speak_text(text)
    cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv.imshow('Image', img)
    cv.imshow('Depth Map', depth_map)
    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows()
cap.release()





