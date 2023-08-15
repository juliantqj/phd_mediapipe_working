# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:22:10 2022

This is mediapipe multipose with Pose (faster)

@author: komarjo1

"""
# %reset -f

import cv2
import mediapipe as mp
import csv
import json
import pandas as pd
import numpy as np
from numpy import linspace
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import signal

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


###

# Name of the file
file_name = 'tt_trial'
video_format = '.mp4'

# If you want to export the dat/video
export_data = 2   # 2 to export the data, 0 to avoid
onefileperframe = 2  # 2 if the export has to be 1 file per frame, 0 if not / to use if the video is long
export_vid = 2   # 2 to export the data, 0 to avoid

# detection parameters
alpha = 0.2  # minimum confidence level from the person-detection model for the detection to be considered successful
beta = 0.9  # Setting it to a higher value can increase robustness of the solution, at the expense of a higher latency
complexity = 2  # complexity from 0 to 2  (more time but better accuracy)
static = False  # If set to false, the solution treats the input images as a video stream

# If you want to mask the video = play with X and Y (1 and 1 means no mask)
MASK = 1  # 1 for the mask, for full video, 0 no mask
left = 0   # how much to remove from left side (in %)
right = 0   # how much to remove from right side (in %)
top = 0   # how much to remove from top (in %)
bottom = 0   # how much to remove from bottom (in %)

#

# file_list = ['videos/im1.png']
# file = 'videos/im1.png'

# For static images:
# with mp_holistic.Holistic(
#    static_image_mode=True,
#    model_complexity=2) as holistic:
#  for idx, file in enumerate(file_list):
#    image = cv2.imread(file)
#    image_height, image_width, _ = image.shape
#    # Convert the BGR image to RGB before processing.
#    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))#
#
#    if results.pose_landmarks:
#      print(
#          f'Nose coordinates: ('
#          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
#          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
#      )
#    # Draw pose, left and right hands, and face landmarks on the image.
#    annotated_image = image.copy()
#    mp_drawing.draw_landmarks(
#        annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
#    mp_drawing.draw_landmarks(
#        annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#    mp_drawing.draw_landmarks(
#        annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#    mp_drawing.draw_landmarks(
#        annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
# cap = cv2.VideoCapture(0)


# For video file input:
cap = cv2.VideoCapture('videos/' + file_name + video_format)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# set the export parameters of the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Export path for the videos    
dirVID = 'videos/export_vid/' + file_name + '/'
# Export path for the coordinates
dirDAT = 'videos/export_dat/' + file_name + '/'

if not os.path.exists(dirDAT):
    os.makedirs(dirDAT)
if not os.path.exists(dirVID):
    os.makedirs(dirVID)

# Create outcome video paths and names
if export_vid > 1:
    out = cv2.VideoWriter((dirVID + file_name + '_Pose_output.mp4'),
                          fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    out_black = cv2.VideoWriter((dirVID + file_name + '_Pose_output_black.mp4'),
                                fourcc, fps, (int(cap.get(3)), int(cap.get(4))))


# Create the mask
if MASK == 1:
    width = cap.get(cv2. CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2. CAP_PROP_FRAME_HEIGHT)
    masks = np.zeros([int(height), int(width)], dtype='uint8')
    topleftcorner = (int(left/100*width), int(top/100*height))
    bottomrightcorner = (int(width-right/100*width), int(height-bottom/100*height))
    masks = cv2.rectangle(masks, topleftcorner, bottomrightcorner, (255, 255, 255), -1)

    # cv2.imshow('test', masks)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # # black is to block, white is to view


# clean the data if a previous version exists already
filePath_temp = (dirDAT + file_name + '_Pose_out_dat.csv')
# As file at filePath is deleted now, so we should check if file exists or not before deleting them
if os.path.exists(filePath_temp):
    os.remove(filePath_temp)


with mp_pose.Pose(        
    static_image_mode=static,  # If set to false, the solution treats the input images as a video stream
    min_detection_confidence=alpha,  # from the person-detection model for the detection to be considered successful
    min_tracking_confidence=beta,  # Setting it to a higher value can increase robustness of the solution,
        # at the expense of a higher latency complexity from 0 to 2  (more time but better accuracy)
    model_complexity=complexity,  # Landmark accuracy as well as inference latency generally go up with the
        # model complexity smooth_landmarks=True) as holistic:
    smooth_landmarks=True) as pose:
  while cap.isOpened():
    frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print('Processed Frame ',int(frame_num))
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    
    image_cut = cv2.bitwise_and(image,image,mask=masks)
    
#    results = holistic.process(image_cut)
    results = pose.process(image_cut)
    
        # Draw landmark annotation on the image
    image.flags.writeable = True
    # change the color scheme
#    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#    image = np.copy(image)
    image_asli = np.zeros(image.shape)
#   mp_drawing.draw_landmarks(
#        image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
#    mp_drawing.draw_landmarks(
#        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#    mp_drawing.draw_landmarks(
#        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
 #       image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    mp_drawing.draw_landmarks(
#        image_asli, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) 
        image_asli, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    

    
    if (export_data > 1):
        if results.pose_world_landmarks:
            landmark_list_pose = results.pose_world_landmarks.landmark
                 
            posisi_x_pose = np.zeros(len(landmark_list_pose))
            posisi_y_pose = np.zeros(len(landmark_list_pose))
            posisi_z_pose = np.zeros(len(landmark_list_pose))
            posisi_vis_pose = np.zeros(len(landmark_list_pose))
            for i in range(len(landmark_list_pose)):
                posisi_x_pose[i] = landmark_list_pose[i].x #* image_asli.shape[1]
                posisi_y_pose[i] = landmark_list_pose[i].y #* image_asli.shape[0]
                posisi_z_pose[i] = landmark_list_pose[i].z #* image_asli.shape[1]
                posisi_vis_pose[i] = landmark_list_pose[i].visibility #* image_asli.shape[1]
                 
#        if results.left_hand_landmarks:
#            landmark_list_left_hand = results.left_hand_landmarks.landmark
#                
#            posisi_x_left_hand = np.zeros(len(landmark_list_left_hand))
#            posisi_y_left_hand = np.zeros(len(landmark_list_left_hand))
#            posisi_z_left_hand = np.zeros(len(landmark_list_left_hand))
#            for i in range(len(landmark_list_left_hand)):
#                posisi_x_left_hand[i] = landmark_list_left_hand[i].x #* image_asli.shape[1]
#                posisi_y_left_hand[i] = landmark_list_left_hand[i].y #* image_asli.shape[0]
#                posisi_z_left_hand[i] = landmark_list_left_hand[i].z #* image_asli.shape[0]
#                
#        if results.right_hand_landmarks:
#            landmark_list_right_hand = results.right_hand_landmarks.landmark
#                
#            posisi_x_right_hand = np.zeros(len(landmark_list_right_hand))
#            posisi_y_right_hand = np.zeros(len(landmark_list_right_hand))
#            posisi_z_right_hand = np.zeros(len(landmark_list_right_hand))
#            for i in range(len(landmark_list_right_hand)):
#                posisi_x_right_hand[i] = landmark_list_right_hand[i].x #* image_asli.shape[1]
#                posisi_y_right_hand[i] = landmark_list_right_hand[i].y #* image_asli.shape[0]
#                posisi_z_right_hand[i] = landmark_list_right_hand[i].z #* image_asli.shape[0]

        
#        if results.face_landmarks:
#            landmark_list_face = results.face_landmarks.landmark
#                
#            posisi_x_face = np.zeros(len(landmark_list_face))
#            posisi_y_face = np.zeros(len(landmark_list_face))
#            posisi_z_face = np.zeros(len(landmark_list_face))
#            for i in range(len(landmark_list_face)):
#                posisi_x_face[i] = landmark_list_face[i].x #* image_asli.shape[1]
#                posisi_y_face[i] = landmark_list_face[i].y #* image_asli.shape[0]
#                posisi_z_face[i] = landmark_list_face[i].z #* image_asli.shape[0]


    if (export_data > 1 and onefileperframe > 1):


            if results.pose_world_landmarks:
                data_teemp=np.concatenate((np.array([posisi_x_pose]),np.array([posisi_y_pose]),np.array([posisi_z_pose]),np.array([posisi_vis_pose])),axis=1) 
                data = np.concatenate((data_teemp,[[width]],[[height]],[[frame_num]]),axis=1) 
                df = pd.DataFrame(data)
                df.rename(columns = {0: 'nose_X', 1: 'left_eye_inner_X',2: 'left_eye_X' ,3: 'left_eye_outer_X' ,4: 'right_eye_inner_X' ,5: 'right_eye_X' ,6: 'right_eye_outer_X' ,7: 'left_ear_X' ,8: 'right_ear_inner_X' ,9: 'mouth_left_X' , 10: 'mouth_right_X',11: 'left_shoulder_X' ,12: 'right_shoulder_X' ,
                         13: 'left_elbow_X',14: 'right_elbow_X' ,15: 'left_wrist_X' ,16: 'right_wrist_X' ,17: 'left_pinky_X' ,18: 'right_pinky_X' ,19: 'left_index_X' ,20: 'right_index_X' ,21: 'left_thumb_X' ,22: 'right_thumb_X' ,23: 'left_hip_X',
                         24: 'right_hip_X',25: 'left_knee_X',26: 'right_knee_X',27: 'left_ankle_X',28: 'right_ankle_X',29: 'left_heel_X',30: 'right_heel_X',31: 'left_foot_index_X',32: 'right_foot_index_X',
                         33: 'nose_Y', 34: 'left_eye_inner_Y',35: 'left_eye_Y' ,36: 'left_eye_outer_Y' ,37: 'right_eye_inner_Y' ,38: 'right_eye_Y' ,39: 'right_eye_outer_Y' ,40: 'left_ear_Y' ,41: 'right_ear_inner_Y' ,42: 'mouth_left_Y' , 43: 'mouth_right_Y',44: 'left_shoulder_Y' ,45: 'right_shoulder_Y' ,
                         46: 'left_elbow_Y',47: 'right_elbow_Y' ,48: 'left_wrist_Y' ,49: 'right_wrist_Y' ,50: 'left_pinky_Y' ,51: 'right_pinky_Y' ,52: 'left_index_Y' ,53: 'right_index_Y' ,54: 'left_thumb_Y' ,55: 'right_thumb_Y' ,56: 'left_hip_Y',
                         57: 'right_hip_Y',58: 'left_knee_Y',59: 'right_knee_Y',60: 'left_ankle_Y',61: 'right_ankle_Y',62: 'left_heel_Y',63: 'right_heel_Y',64: 'left_foot_index_Y',65: 'right_foot_index_Y',
                         66: 'nose_Z', 67: 'left_eye_inner_Z',68: 'left_eye_Z' ,69: 'left_eye_outer_Z' ,70: 'right_eye_inner_Z' ,71: 'right_eye_Z' ,72: 'right_eye_outer_Z' ,73: 'left_ear_Z' ,74: 'right_ear_inner_Z' ,75: 'mouth_left_Z' , 76: 'mouth_right_Z',77: 'left_shoulder_Z' ,78: 'right_shoulder_Z' ,
                         79: 'left_elbow_Z',80: 'right_elbow_Z' ,81: 'left_wrist_Z' ,82: 'right_wrist_Z' ,83: 'left_pinky_Z' ,84: 'right_pinky_Z' ,85: 'left_index_Z' ,86: 'right_index_Z' ,87: 'left_thumb_Z' ,88: 'right_thumb_Z' ,89: 'left_hip_Z',
                         90: 'right_hip_Z',91: 'left_knee_Z',92: 'right_knee_Z',93: 'left_ankle_Z',94: 'right_ankle_Z',95: 'left_heel_Z',96: 'right_heel_Z',97: 'left_foot_index_Z',98: 'right_foot_index_Z', 
                         99: 'nose_vis', 100: 'left_eye_inner_vis',101: 'left_eye_vis' ,102: 'left_eye_outer_vis' ,103: 'right_eye_inner_vis' ,104: 'right_eye_vis' ,105: 'right_eye_outer_vis' ,106: 'left_ear_vis' ,107: 'right_ear_inner_vis' ,108: 'mouth_left_vis' , 109: 'mouth_right_vis',110: 'left_shoulder_vis' ,111: 'right_shoulder_vis' ,
                         112: 'left_elbow_vis',113: 'right_elbow_vis' ,114: 'left_wrist_vis' ,115: 'right_wrist_vis' ,116: 'left_pinky_vis' ,117: 'right_pinky_vis' ,118: 'left_index_vis' ,119: 'right_index_vis' ,120: 'left_thumb_vis' ,121: 'right_thumb_vis' ,122: 'left_hip_vis',
                         123: 'right_hip_vis',124: 'left_knee_vis',125: 'right_knee_vis',126: 'left_ankle_vis',127: 'right_ankle_vis',128: 'left_heel_vis',129: 'right_heel_vis',130: 'left_foot_index_vis',131: 'right_foot_index_vis', 
                         132: 'width_X', 133: 'height_Y', 134: 'frame_num'
                         }, inplace = True)      
            else: 
                data_teemp=  np.zeros((1,132), dtype=int)     # np.concatenate((np.array([posisi_x_pose]),np.array([posisi_y_pose]),np.array([posisi_z_pose])),axis=1) 
                data = np.concatenate((data_teemp,[[width]],[[height]],[[frame_num]]),axis=1)
                df = pd.DataFrame(data)
                df.rename(columns = {0: 'nose_X', 1: 'left_eye_inner_X',2: 'left_eye_X' ,3: 'left_eye_outer_X' ,4: 'right_eye_inner_X' ,5: 'right_eye_X' ,6: 'right_eye_outer_X' ,7: 'left_ear_X' ,8: 'right_ear_inner_X' ,9: 'mouth_left_X' , 10: 'mouth_right_X',11: 'left_shoulder_X' ,12: 'right_shoulder_X' ,
                         13: 'left_elbow_X',14: 'right_elbow_X' ,15: 'left_wrist_X' ,16: 'right_wrist_X' ,17: 'left_pinky_X' ,18: 'right_pinky_X' ,19: 'left_index_X' ,20: 'right_index_X' ,21: 'left_thumb_X' ,22: 'right_thumb_X' ,23: 'left_hip_X',
                         24: 'right_hip_X',25: 'left_knee_X',26: 'right_knee_X',27: 'left_ankle_X',28: 'right_ankle_X',29: 'left_heel_X',30: 'right_heel_X',31: 'left_foot_index_X',32: 'right_foot_index_X',
                         33: 'nose_Y', 34: 'left_eye_inner_Y',35: 'left_eye_Y' ,36: 'left_eye_outer_Y' ,37: 'right_eye_inner_Y' ,38: 'right_eye_Y' ,39: 'right_eye_outer_Y' ,40: 'left_ear_Y' ,41: 'right_ear_inner_Y' ,42: 'mouth_left_Y' , 43: 'mouth_right_Y',44: 'left_shoulder_Y' ,45: 'right_shoulder_Y' ,
                         46: 'left_elbow_Y',47: 'right_elbow_Y' ,48: 'left_wrist_Y' ,49: 'right_wrist_Y' ,50: 'left_pinky_Y' ,51: 'right_pinky_Y' ,52: 'left_index_Y' ,53: 'right_index_Y' ,54: 'left_thumb_Y' ,55: 'right_thumb_Y' ,56: 'left_hip_Y',
                         57: 'right_hip_Y',58: 'left_knee_Y',59: 'right_knee_Y',60: 'left_ankle_Y',61: 'right_ankle_Y',62: 'left_heel_Y',63: 'right_heel_Y',64: 'left_foot_index_Y',65: 'right_foot_index_Y',
                         66: 'nose_Z', 67: 'left_eye_inner_Z',68: 'left_eye_Z' ,69: 'left_eye_outer_Z' ,70: 'right_eye_inner_Z' ,71: 'right_eye_Z' ,72: 'right_eye_outer_Z' ,73: 'left_ear_Z' ,74: 'right_ear_inner_Z' ,75: 'mouth_left_Z' , 76: 'mouth_right_Z',77: 'left_shoulder_Z' ,78: 'right_shoulder_Z' ,
                         79: 'left_elbow_Z',80: 'right_elbow_Z' ,81: 'left_wrist_Z' ,82: 'right_wrist_Z' ,83: 'left_pinky_Z' ,84: 'right_pinky_Z' ,85: 'left_index_Z' ,86: 'right_index_Z' ,87: 'left_thumb_Z' ,88: 'right_thumb_Z' ,89: 'left_hip_Z',
                         90: 'right_hip_Z',91: 'left_knee_Z',92: 'right_knee_Z',93: 'left_ankle_Z',94: 'right_ankle_Z',95: 'left_heel_Z',96: 'right_heel_Z',97: 'left_foot_index_Z',98: 'right_foot_index_Z', 
                         99: 'nose_vis', 100: 'left_eye_inner_vis',101: 'left_eye_vis' ,102: 'left_eye_outer_vis' ,103: 'right_eye_inner_vis' ,104: 'right_eye_vis' ,105: 'right_eye_outer_vis' ,106: 'left_ear_vis' ,107: 'right_ear_inner_vis' ,108: 'mouth_left_vis' , 109: 'mouth_right_vis',110: 'left_shoulder_vis' ,111: 'right_shoulder_vis' ,
                         112: 'left_elbow_vis',113: 'right_elbow_vis' ,114: 'left_wrist_vis' ,115: 'right_wrist_vis' ,116: 'left_pinky_vis' ,117: 'right_pinky_vis' ,118: 'left_index_vis' ,119: 'right_index_vis' ,120: 'left_thumb_vis' ,121: 'right_thumb_vis' ,122: 'left_hip_vis',
                         123: 'right_hip_vis',124: 'left_knee_vis',125: 'right_knee_vis',126: 'left_ankle_vis',127: 'right_ankle_vis',128: 'left_heel_vis',129: 'right_heel_vis',130: 'left_foot_index_vis',131: 'right_foot_index_vis', 
                         132: 'width_X', 133: 'height_Y', 134: 'frame_num'
                         }, inplace = True)       

        # This one to export 1 file per row

            df.to_csv((dirDAT + file_name + '_Pose_Frame ' + str(int(frame_num)) + '.csv'), mode='w', sep=',', index=False, header=True)    
 #       df.to_json((dirDAT + file_name + '_Pose_Frame ' + str(int(frame_num)) + '.json'), index=True)       


    # Export one file with all frames
 #   if (export_data > 1 and onefileperframe < 1):
        
 #       data_teemp=np.concatenate((np.array([posisi_x_pose]),np.array([posisi_y_pose]),np.array([posisi_z_pose])),axis=1) 
 #       data = np.concatenate((data_teemp,[[width]],[[height]],[[frame_num]]),axis=1) 
 #       df = pd.DataFrame(data)
 #       df.rename(columns = {0: 'nose_X', 1: 'left_eye_inner_X',2: 'left_eye_X' ,3: 'left_eye_outer_X' ,4: 'right_eye_inner_X' ,5: 'right_eye_X' ,6: 'right_eye_outer_X' ,7: 'left_ear_X' ,8: 'right_ear_inner_X' ,9: 'mouth_left_X' , 10: 'mouth_right_X',11: 'left_shoulder_X' ,12: 'right_shoulder_X' ,
 #                        13: 'left_elbow_X',14: 'right_elbow_X' ,15: 'left_wrist_X' ,16: 'right_wrist_X' ,17: 'left_pinky_X' ,18: 'right_pinky_X' ,19: 'left_index_X' ,20: 'right_index_X' ,21: 'left_thumb_X' ,22: 'right_thumb_X' ,23: 'left_hip_X',
 #                        24: 'right_hip_X',25: 'left_knee_X',26: 'right_knee_X',27: 'left_ankle_X',28: 'right_ankle_X',29: 'left_heel_X',30: 'right_heel_X',31: 'left_foot_index_X',32: 'right_foot_index_X',
 #                        33: 'nose_Y', 34: 'left_eye_inner_Y',35: 'left_eye_Y' ,36: 'left_eye_outer_Y' ,37: 'right_eye_inner_Y' ,38: 'right_eye_Y' ,39: 'right_eye_outer_Y' ,40: 'left_ear_Y' ,41: 'right_ear_inner_Y' ,42: 'mouth_left_Y' , 43: 'mouth_right_Y',44: 'left_shoulder_Y' ,45: 'right_shoulder_Y' ,
 #                        46: 'left_elbow_Y',47: 'right_elbow_Y' ,48: 'left_wrist_Y' ,49: 'right_wrist_Y' ,50: 'left_pinky_Y' ,51: 'right_pinky_Y' ,52: 'left_index_Y' ,53: 'right_index_Y' ,54: 'left_thumb_Y' ,55: 'right_thumb_Y' ,56: 'left_hip_Y',
 #                        57: 'right_hip_Y',58: 'left_knee_Y',59: 'right_knee_Y',60: 'left_ankle_Y',61: 'right_ankle_Y',62: 'left_heel_Y',63: 'right_heel_Y',64: 'left_foot_index_Y',65: 'right_foot_index_Y',
 #                        66: 'nose_Z', 67: 'left_eye_inner_Z',68: 'left_eye_Z' ,69: 'left_eye_outer_Z' ,70: 'right_eye_inner_Z' ,71: 'right_eye_Z' ,72: 'right_eye_outer_Z' ,73: 'left_ear_Z' ,74: 'right_ear_inner_Z' ,75: 'mouth_left_Z' , 76: 'mouth_right_Z',77: 'left_shoulder_Z' ,78: 'right_shoulder_Z' ,
 #                        79: 'left_elbow_Z',80: 'right_elbow_Z' ,81: 'left_wrist_Z' ,82: 'right_wrist_Z' ,83: 'left_pinky_Y' ,84: 'right_pinky_Z' ,85: 'left_index_Z' ,86: 'right_index_Z' ,87: 'left_thumb_Z' ,88: 'right_thumb_Z' ,89: 'left_hip_Z',
 #                        90: 'right_hip_Z',91: 'left_knee_Z',92: 'right_knee_Z',93: 'left_ankle_Z',94: 'right_ankle_Z',95: 'left_heel_Z',96: 'right_heel_Y',97: 'left_foot_index_Z',98: 'right_foot_index_Z', 99: 'width_X', 100: 'height_Y', 101: 'frame_num'
 #                        }, inplace = True)   
 #
 #       
 #       df.to_csv((dirDAT + file_name +'_Pose_out_dat.csv'), mode='a', sep=',', index=False, header=False)    



    
        # create the black background 
    image_out = image_asli
    image_out = image_out / image_out.max()  # normalizes data in range 0 - 255
    image_out = 255 * image_out
    img_out = image_out.astype(np.uint8)
    
    
    
    if (export_vid > 1): 
    # write the frame and export the mp4 file with a black background
        out_black.write(img_out) 
    # write the frame and export the mp4 file with the original video as background
        out.write(image)
    
    
    
    # remove if you don't want to see the video
    cv2.imshow('MediaPipe Pose', image)

    if cv2.waitKey(5) & 0xFF == ord('d'):
      break


    if (export_data > 1 and onefileperframe < 1):
    # read the exported data and export one file and replace the headers
        temp = pd.read_csv((dirDAT + file_name +'_Pose_out_dat.csv'), sep=',', header=None)
        temp.rename(columns = {0: 'nose_X', 1: 'left_eye_inner_X',2: 'left_eye_X' ,3: 'left_eye_outer_X' ,4: 'right_eye_inner_X' ,5: 'right_eye_X' ,6: 'right_eye_outer_X' ,7: 'left_ear_X' ,8: 'right_ear_inner_X' ,9: 'mouth_left_X' , 10: 'mouth_right_X',11: 'left_shoulder_X' ,12: 'right_shoulder_X' ,
                         13: 'left_elbow_X',14: 'right_elbow_X' ,15: 'left_wrist_X' ,16: 'right_wrist_X' ,17: 'left_pinky_X' ,18: 'right_pinky_X' ,19: 'left_index_X' ,20: 'right_index_X' ,21: 'left_thumb_X' ,22: 'right_thumb_X' ,23: 'left_hip_X',
                         24: 'right_hip_X',25: 'left_knee_X',26: 'right_knee_X',27: 'left_ankle_X',28: 'right_ankle_X',29: 'left_heel_X',30: 'right_heel_X',31: 'left_foot_index_X',32: 'right_foot_index_X',
                         33: 'nose_Y', 34: 'left_eye_inner_Y',35: 'left_eye_Y' ,36: 'left_eye_outer_Y' ,37: 'right_eye_inner_Y' ,38: 'right_eye_Y' ,39: 'right_eye_outer_Y' ,40: 'left_ear_Y' ,41: 'right_ear_inner_Y' ,42: 'mouth_left_Y' , 43: 'mouth_right_Y',44: 'left_shoulder_Y' ,45: 'right_shoulder_Y' ,
                         46: 'left_elbow_Y',47: 'right_elbow_Y' ,48: 'left_wrist_Y' ,49: 'right_wrist_Y' ,50: 'left_pinky_Y' ,51: 'right_pinky_Y' ,52: 'left_index_Y' ,53: 'right_index_Y' ,54: 'left_thumb_Y' ,55: 'right_thumb_Y' ,56: 'left_hip_Y',
                         57: 'right_hip_Y',58: 'left_knee_Y',59: 'right_knee_Y',60: 'left_ankle_Y',61: 'right_ankle_Y',62: 'left_heel_Y',63: 'right_heel_Y',64: 'left_foot_index_Y',65: 'right_foot_index_Y',
                         66: 'nose_Z', 67: 'left_eye_inner_Z',68: 'left_eye_Z' ,69: 'left_eye_outer_Z' ,70: 'right_eye_inner_Z' ,71: 'right_eye_Z' ,72: 'right_eye_outer_Z' ,73: 'left_ear_Z' ,74: 'right_ear_inner_Z' ,75: 'mouth_left_Z' , 76: 'mouth_right_Z',77: 'left_shoulder_Z' ,78: 'right_shoulder_Z' ,
                         79: 'left_elbow_Z',80: 'right_elbow_Z' ,81: 'left_wrist_Z' ,82: 'right_wrist_Z' ,83: 'left_pinky_Z' ,84: 'right_pinky_Z' ,85: 'left_index_Z' ,86: 'right_index_Z' ,87: 'left_thumb_Z' ,88: 'right_thumb_Z' ,89: 'left_hip_Z',
                         90: 'right_hip_Z',91: 'left_knee_Z',92: 'right_knee_Z',93: 'left_ankle_Z',94: 'right_ankle_Z',95: 'left_heel_Z',96: 'right_heel_Z',97: 'left_foot_index_Z',98: 'right_foot_index_Z', 
                         99: 'nose_vis', 100: 'left_eye_inner_vis',101: 'left_eye_vis' ,102: 'left_eye_outer_vis' ,103: 'right_eye_inner_vis' ,104: 'right_eye_vis' ,105: 'right_eye_outer_vis' ,106: 'left_ear_vis' ,107: 'right_ear_inner_vis' ,108: 'mouth_left_vis' , 109: 'mouth_right_vis',110: 'left_shoulder_vis' ,111: 'right_shoulder_vis' ,
                         112: 'left_elbow_vis',113: 'right_elbow_vis' ,114: 'left_wrist_vis' ,115: 'right_wrist_vis' ,116: 'left_pinky_vis' ,117: 'right_pinky_vis' ,118: 'left_index_vis' ,119: 'right_index_vis' ,120: 'left_thumb_vis' ,121: 'right_thumb_vis' ,122: 'left_hip_vis',
                         123: 'right_hip_vis',124: 'left_knee_vis',125: 'right_knee_vis',126: 'left_ankle_vis',127: 'right_ankle_vis',128: 'left_heel_vis',129: 'right_heel_vis',130: 'left_foot_index_vis',131: 'right_foot_index_vis', 
                         132: 'width_X', 133: 'height_Y', 134: 'frame_num'
                         }, inplace = True)   
        temp.to_csv((dirDAT + file_name +'_Pose_out_dat.csv'), mode='w', sep=',', index=True, header=True)  
        temp.to_json((dirDAT + file_name +'_Pose_out_dat.json'), index=True)  
      
    
cap.release()
out.release()
out_black.release()

fig = plt.figure(figsize = (8,8))
ax = plt.axes(projection = '3d')
ax.plot3D(x, a, **next(color_cycle))
for angle in range(0, 360):
    ax.view_init(angle, 30)
    plt.draw()
    plt.pause(.001)

plt.show()

cv2.destroyAllWindows()  

cv2.waitKey(1)
    

if (export_data > 1 and onefileperframe > 1):
    # lire tous les csv dans ce dossier
        os.chdir('/Users/juliantanquahjian/mediapipe_working/videos/export_dat/' + file_name)
        extension = 'csv'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    #combine all files in the list
        temp = pd.concat([pd.read_csv(f) for f in all_filenames ], axis=0)
        temp = temp.sort_values(by=['frame_num'],axis=0,ascending=True)
        # mettre les header
#        df.rename(columns = {0: 'nose_X', 1: 'left_eye_inner_X',2: 'left_eye_X' ,3: 'left_eye_outer_X' ,4: 'right_eye_inner_X' ,5: 'right_eye_X' ,6: 'right_eye_outer_X' ,7: 'left_ear_X' ,8: 'right_ear_inner_X' ,9: 'mouth_left_X' , 10: 'mouth_right_X',11: 'left_shoulder_X' ,12: 'right_shoulder_X' ,
#                         13: 'left_elbow_X',14: 'right_elbow_X' ,15: 'left_wrist_X' ,16: 'right_wrist_X' ,17: 'left_pinky_X' ,18: 'right_pinky_X' ,19: 'left_index_X' ,20: 'right_index_X' ,21: 'left_thumb_X' ,22: 'right_thumb_X' ,23: 'left_hip_X',
#                         24: 'right_hip_X',25: 'left_knee_X',26: 'right_knee_X',27: 'left_ankle_X',28: 'right_ankle_X',29: 'left_heel_X',30: 'right_heel_X',31: 'left_foot_index_X',32: 'right_foot_index_X',
#                         33: 'nose_Y', 34: 'left_eye_inner_Y',35: 'left_eye_Y' ,36: 'left_eye_outer_Y' ,37: 'right_eye_inner_Y' ,38: 'right_eye_Y' ,39: 'right_eye_outer_Y' ,40: 'left_ear_Y' ,41: 'right_ear_inner_Y' ,42: 'mouth_left_Y' , 43: 'mouth_right_Y',44: 'left_shoulder_Y' ,45: 'right_shoulder_Y' ,
#                         46: 'left_elbow_Y',47: 'right_elbow_Y' ,48: 'left_wrist_Y' ,49: 'right_wrist_Y' ,50: 'left_pinky_Y' ,51: 'right_pinky_Y' ,52: 'left_index_Y' ,53: 'right_index_Y' ,54: 'left_thumb_Y' ,55: 'right_thumb_Y' ,56: 'left_hip_Y',
#                         57: 'right_hip_Y',58: 'left_knee_Y',59: 'right_knee_Y',60: 'left_ankle_Y',61: 'right_ankle_Y',62: 'left_heel_Y',63: 'right_heel_Y',64: 'left_foot_index_Y',65: 'right_foot_index_Y',
#                         66: 'nose_Z', 67: 'left_eye_inner_Z',68: 'left_eye_Z' ,69: 'left_eye_outer_Z' ,70: 'right_eye_inner_Z' ,71: 'right_eye_Z' ,72: 'right_eye_outer_Z' ,73: 'left_ear_Z' ,74: 'right_ear_inner_Z' ,75: 'mouth_left_Z' , 76: 'mouth_right_Z',77: 'left_shoulder_Z' ,78: 'right_shoulder_Z' ,
#                         79: 'left_elbow_Z',80: 'right_elbow_Z' ,81: 'left_wrist_Z' ,82: 'right_wrist_Z' ,83: 'left_pinky_Y' ,84: 'right_pinky_Z' ,85: 'left_index_Z' ,86: 'right_index_Z' ,87: 'left_thumb_Z' ,88: 'right_thumb_Z' ,89: 'left_hip_Z',
#                         90: 'right_hip_Z',91: 'left_knee_Z',92: 'right_knee_Z',93: 'left_ankle_Z',94: 'right_ankle_Z',95: 'left_heel_Z',96: 'right_heel_Y',97: 'left_foot_index_Z',98: 'right_foot_index_Z', 99: 'width_X', 100: 'height_Y', 101: 'frame_num'
#                         }, inplace = True)   
 
        if not os.path.exists(("combined/")):
            os.makedirs("combined/")
            
        temp.reset_index(inplace=True)
        temp.to_csv(("combined/" + file_name +'_Pose_out_dat.csv'), mode='w', sep=',', index=True, header=True)  
        temp.to_json(("combined/" + file_name +'_Pose_out_dat.json'), index=True)  
    


    
    
    