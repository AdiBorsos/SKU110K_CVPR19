# import keras_retinanet
from object_detector_retinanet.keras_retinanet import models
from object_detector_retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from object_detector_retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption, draw_annotations, draw_detections
from object_detector_retinanet.keras_retinanet.utils.colors import label_color

# import for EM Merger and viz
from object_detector_retinanet.keras_retinanet.utils import EmMerger
from object_detector_retinanet.utils import create_folder, root_dir


# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import datetime

import csv
# set tf backend to allow memory to grow, instead of claiming everything

from keras import backend as K
import tensorflow as tf
from PIL import Image
from object_detector_retinanet.keras_retinanet.preprocessing.csv_generator import CSVGenerator

hard_score_rate = 0.5
# for filtering predictions based on score (objectness/confidence)
threshold = 0.3
score_threshold=0.05
max_detections=9999

model_path = os.path.abspath('D:/Tech assesments/_CV/SKU110K_CodeGit/object_detector_retinanet/iou_resnet50_csv_06.h5')
model = models.load_model(model_path, backbone_name='resnet50', convert=True)
    
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
    everyX = 10

    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 

        if (count % everyX == 0):
            # Saves the frames with frame-count 
            cv2.imwrite("D:/Tech assesments/_CV/SKU110K_CodeGit/images/frame%d.jpg" % count, image) 
  
        count += 1
        print (count)

def PredThisPath(model = None, videoPath = None):
    csv_data_lst = []
    csv_data_lst.append(['image_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'hard_score'])
    result_dir = os.path.join(root_dir(), 'results')
    create_folder(result_dir)
    timestamp = datetime.datetime.utcnow()
    res_file = result_dir + '/detections_output_iou_{}_{}.csv'.format(hard_score_rate, timestamp.strftime("%c").replace(" ", "_").replace(":", "_"))
    save_path = result_dir + '/saveDir'
    
    #image_p = os.path.abspath('D:/Tech assesments/_CV/SKU110K_CodeGit/images/test_01.jpg')
    #image_p = os.path.abspath(image_path)

    cap = cv2.VideoCapture(videoPath)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float 
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('D:/Tech assesments/_CV/output.mp4', fourcc, fps, (width, height))
    
    image_name = 'frame{}'.format(0)
    
    maxFrames = 100
    predFramerate = 10
    frameNo = -1
    success = 1
    while(success):
        frameNo = frameNo + 1

        # read the frames
        success, frame = cap.read()

        if (frameNo % predFramerate != 0):
            for row in csv_data_lst:
                if (row[0] == image_name):
                    cv2.rectangle(frame, (row[1], row[2]), (row[3], row[4]), (0, 0, 255), 2, cv2.LINE_AA)
            
            out.write(frame)
            continue

        # #A line
        # cv2.line(frame, (500, 400), (640, 480),(0,255,0), 3)
        # cv2.putText(frame, "test!",(105, 105),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(225,0,0))
        # out.write(frame)

        image_name =  'frame{}'.format(frameNo) 
        image = frame # not sure if this is BGR or not 
        
        raw_image = frame # not sure read_image_bgr(image_p)

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)


        image, scale = resize_image(image)


        """print(*image)"""

        # Run inference
        boxes, hard_scores, labels, soft_scores = model.predict_on_batch(np.expand_dims(image, axis=0))
        soft_scores = np.squeeze(soft_scores, axis=-1)
        soft_scores = hard_score_rate * hard_scores + (1 - hard_score_rate) * soft_scores
        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(hard_scores[0, :] > score_threshold)[0]

        # select those scores
        scores = soft_scores[0][indices]
        hard_scores = hard_scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_hard_scores = hard_scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        results = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_hard_scores, axis=1),
                np.expand_dims(image_labels, axis=1)], axis=1)
        filtered_data = EmMerger.merge_detections(image_name, results, frame)
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for _, detection in filtered_data.iterrows():
            box = np.asarray([detection['x1'], detection['y1'], detection['x2'], detection['y2']])

            cv2.rectangle(frame, (detection['x1'], detection['y1']), (detection['x2'], detection['y2']), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, str(detection['confidence']), (detection['x1'], detection['y1'] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

            filtered_boxes.append(box)
            filtered_scores.append(detection['confidence'])
            filtered_labels.append('{0:.2f}'.format(detection['hard_score']))
            row = [image_name, detection['x1'], detection['y1'], detection['x2'], detection['y2'],
                    detection['confidence'], detection['hard_score']]
            csv_data_lst.append(row)
        
        cv2.imwrite(os.path.join("D:/Tech assesments/_CV/frames", '{}.png'.format(image_name)), frame)
        out.write(frame)

        
        if (maxFrames < frameNo and maxFrames != -1):
            success = 0

    out.release()

    # if save_path is not None:
    #     create_folder(save_path)

    #     #draw_annotations(raw_image, generator.load_annotations(0), label_to_name=generator.label_to_name)
    #     draw_detections(raw_image, np.asarray(filtered_boxes), np.asarray(filtered_scores),
    #                     np.asarray(filtered_labels), color=(0, 0, 255))

    #     cv2.imwrite(os.path.join(save_path, '{}.png'.format(image_name)), raw_image)

    # copy detections to all_detections
    # for label in range(generator.num_classes()):
    #     all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]


    # Save annotations csv file
    with open(res_file, 'a+') as fl_csv:
        writer = csv.writer(fl_csv)
        writer.writerows(csv_data_lst)
    print("Saved output.csv file")


PredThisPath(model, "D:/Tech assesments/_CV/Trim1.mp4")


# FrameCapture("D:/Tech assesments/_CV/Trim1.mp4")

# directory = "D:/Tech assesments/_CV/SKU110K_CodeGit/images"

# for subdir, dirs, files in os.walk(directory):
#     for file in files:
#         print(os.path.join(directory, file))

#         PredThisPath(os.path.join(directory, file))


