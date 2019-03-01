'''
Detects pneimonia using a Darknet yolo model.
'''

import cv2 as cv
import os
import numpy as np


confidance_threshold = 0.2
nms_threshold = 0.2

model_config = "yolov3-PneimoniaDetection.cfg"
model_weights = "yolov3-PneimoniaDetection_final.weights"


def postprocess(image, outs):
    '''Processes the output of the model and draws the detection boxes on the images.
    Args:
        image: Image that the boxes will be drawn on.
        outs: Output of the model.
    Returns:
        None
    '''
    image_height = image.shape[0]
    image_width = image.shape[1]
 
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            confidence = detection[5]
            if confidence > confidance_threshold:
                x_center = int(float(detection[0]) * image_width)
                y_center = int(float(detection[1]) * image_height)
                width = int(float(detection[2]) * image_width)
                height = int(float(detection[3]) * image_height)
                left = int(x_center - width / 2)
                top = int(y_center - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confidance_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawBox(image, left, top, left + width, top + height, confidences[i])


def drawBox(image, left, top, right, bottom, conf=None, color=(0,0,255)):
    '''Processes the output of the model and draws the detection boxes on the images.
    Args:
        image: Image that the boxes will be drawn on.
        left: Left side of the box.
        top: Top side of the box.
        right: Right side of the box.
        bottom: Bottom side of the box.
        conf: Confidence to be displayed. String will be shown as is. Number will be maultiplied be 100.
        color: Color of the box to draw. Color is a BRG tuple (Blue, Green, Red).
    Returns:
        None
    '''
    pts = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], np.int32)
    cv.polylines(image, [pts], True, color)
    if conf is not None:
        if isinstance(conf, str):
            label = conf
        else:
            label = "{0:.1f}%".format(conf*100)

        label_size, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv.putText(image, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))


#Setting up model.
net = cv.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

#Opens a window to show the images.
window_name = "PneimoniaDetection"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.resizeWindow(window_name, 1024,1024)

#Reads the file containg the paths of the images to run.
file_path = "images.txt"
image_names = []
with open(file_path, 'r') as file:
    image_names = file.read().rsplit("\n")

for image_name in image_names:
    print(image_name)
    image = cv.imread(image_name, cv.IMREAD_GRAYSCALE)

    #Reads the label file to get the actual detection boxes.
    text_file = os.path.splitext(image_name)[0] + ".txt"
    real_boxes = []
    with open(text_file, 'r') as file:
        lines = file.read().rsplit("\n")
        for line in lines:
            if line is '':
                continue
            x_center, y_center, width, height = line.split()[1:5]
            width = int(float(width) * image.shape[0])
            height = int(float(height) * image.shape[1])
            x_center = float(x_center) * image.shape[0]
            y_center = float(y_center) * image.shape[1]
            left = int(float(x_center) - width / 2)
            top = int(float(y_center) - height / 2)
            right = left + width
            bottom = top + height
            real_boxes.append([left, top, right, bottom])

    #Forward passes the image through the model.
    blob = cv.dnn.blobFromImage(image, 1/255, (416, 416), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    #Draws boxes and actual boxes on the image and shows it.
    color_image = cv.cvtColor(image ,cv.COLOR_GRAY2RGB)
    postprocess(color_image, outs)
    for box in real_boxes:
        drawBox(color_image, box[0], box[1], box[2], box[3], color=(0,255,0), conf="Actual")
    cv.imshow(window_name, color_image)
    cv.waitKey(0)
