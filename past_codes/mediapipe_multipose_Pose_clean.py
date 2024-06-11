import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import glob

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Name of the file
file_name = 'old_short_1'
video_format = '.mp4'

# If you want to export the dat/video
export_data = 2  # 2 to export the data, 0 to avoid
one_file_per_frame = 2  # 2 if the export has to be 1 file per frame, 0 if not / to use if the video is long
export_vid = 2  # 2 to export the data, 0 to avoid

# detection parameters
static = False  # If set to false, the solution treats the input images as a video stream
alpha = 0.2  # minimum confidence level from the person-detection model for the detection to be considered successful
beta = 0.9  # Setting it to a higher value can increase robustness of the solution, at the expense of a higher latency
complexity = 2  # complexity from 0 to 2 (improves landmark accuracy but inference latency goes up, i.e., slower)

# If you want to mask the video = play with X and Y (1 and 1 means no mask)
MASK = 1  # 1 for the mask, for full video, 0 no mask
left = 0  # how much to remove from left side (in %)
right = 0  # how much to remove from right side (in %)
top = 0  # how much to remove from top (in %)
bottom = 0  # how much to remove from bottom (in %)

# For video file input:
cap = cv2.VideoCapture('videos/' + file_name + video_format)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Set export parameters for video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Set export path for  video
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
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    masks = np.zeros([int(height), int(width)], dtype='uint8')
    top_left_corner = (int(left / 100 * width), int(top / 100 * height))
    bottom_right_corner = (int(width - right / 100 * width), int(height - bottom / 100 * height))
    masks = cv2.rectangle(masks, top_left_corner, bottom_right_corner, (255, 255, 255), -1)

    # Uncomment code to test mask size
    # cv2.imshow('test', masks)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # # black is to block, white is to view

# Clean the data if a previous version exists already
filePath_temp = (dirDAT + file_name + '_Pose_out_dat.csv')

# As file at filePath is deleted now, so we should check if file exists or not before deleting them
if os.path.exists(filePath_temp):
    os.remove(filePath_temp)

with mp_pose.Pose(
        # If set to false, the solution treats the input images as a video stream
        static_image_mode=static, min_detection_confidence=alpha, min_tracking_confidence=beta,
        model_complexity=complexity,
        smooth_landmarks=True) as pose:
    while cap.isOpened():
        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print('Processed Frame ', int(frame_num))
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")

            # If loading a video, use 'break' instead of 'continue'
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.

        image.flags.writeable = False

        image_cut = cv2.bitwise_and(image, image, mask=masks)

        results = pose.process(image_cut)

        # Draw landmark annotation on the image
        image.flags.writeable = True
        image_shape_zero = np.zeros(image.shape)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # frame_number_text = "Frame: " + str(int(frame_num))
        # cv2.putText(image, frame_number_text, (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_number_text = "Frame: " + str(int(frame_num))

        # Determine the text size
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 1
        text_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(frame_number_text, text_font, text_scale, text_thickness)

        # Add padding around the text
        background_padding_x = 10  # Adjust the padding around the text horizontally as needed
        background_padding_y = 5  # Adjust the padding around the text vertically as needed

        # Calculate background size based on text size and padding
        background_width = text_width + 2 * background_padding_x
        background_height = text_height + 2 * background_padding_y

        # Add a white background behind the text
        background_color = (255, 255, 255)  # White
        background_position = (50, 50)  # Adjust the background position as needed
        background_start = (background_position[0], background_position[1])
        background_end = (background_start[0] + background_width, background_start[1] + background_height)
        cv2.rectangle(image, background_start, background_end, background_color, -1)

        # Add the frame number text
        text_position = (background_position[0] + background_padding_x, background_position[1]
                         + text_height + background_padding_y)
        text_color = (0, 0, 255)  # Red
        cv2.putText(image, frame_number_text, text_position, text_font, text_scale, text_color, text_thickness,
                    cv2.LINE_AA)

        mp_drawing.draw_landmarks(image_shape_zero, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if export_data > 1:
            if results.pose_world_landmarks:
                landmark_list_pose = results.pose_world_landmarks.landmark

                position_x_pose = np.zeros(len(landmark_list_pose))
                position_y_pose = np.zeros(len(landmark_list_pose))
                position_z_pose = np.zeros(len(landmark_list_pose))
                position_vis_pose = np.zeros(len(landmark_list_pose))
                for i in range(len(landmark_list_pose)):
                    position_x_pose[i] = landmark_list_pose[i].x  # * image_shape_zero.shape[1]
                    position_y_pose[i] = landmark_list_pose[i].y  # * image_shape_zero.shape[0]˙
                    position_z_pose[i] = landmark_list_pose[i].z  # * image_shape_zero.shape[1]
                    position_vis_pose[i] = landmark_list_pose[i].visibility  # * image_shape_zero.shape[1]

        if export_data > 1 and one_file_per_frame > 1:
            if results.pose_world_landmarks:
                data_temp = np.concatenate((np.array([position_x_pose]), np.array([position_y_pose]),
                                            np.array([position_z_pose]), np.array([position_vis_pose])), axis=1)
                data = np.concatenate((data_temp, [[width]], [[height]], [[frame_num]]), axis=1)
                df = pd.DataFrame(data)
                df.rename(columns={0: 'nose_X', 1: 'left_eye_inner_X', 2: 'left_eye_X', 3: 'left_eye_outer_X',
                                   4: 'right_eye_inner_X', 5: 'right_eye_X', 6: 'right_eye_outer_X', 7: 'left_ear_X',
                                   8: 'right_ear_inner_X', 9: 'mouth_left_X', 10: 'mouth_right_X',
                                   11: 'left_shoulder_X', 12: 'right_shoulder_X',
                                   13: 'left_elbow_X', 14: 'right_elbow_X', 15: 'left_wrist_X', 16: 'right_wrist_X',
                                   17: 'left_pinky_X', 18: 'right_pinky_X', 19: 'left_index_X', 20: 'right_index_X',
                                   21: 'left_thumb_X', 22: 'right_thumb_X', 23: 'left_hip_X',
                                   24: 'right_hip_X', 25: 'left_knee_X', 26: 'right_knee_X', 27: 'left_ankle_X',
                                   28: 'right_ankle_X', 29: 'left_heel_X', 30: 'right_heel_X', 31: 'left_foot_index_X',
                                   32: 'right_foot_index_X',
                                   33: 'nose_Y', 34: 'left_eye_inner_Y', 35: 'left_eye_Y', 36: 'left_eye_outer_Y',
                                   37: 'right_eye_inner_Y', 38: 'right_eye_Y', 39: 'right_eye_outer_Y',
                                   40: 'left_ear_Y', 41: 'right_ear_inner_Y', 42: 'mouth_left_Y', 43: 'mouth_right_Y',
                                   44: 'left_shoulder_Y', 45: 'right_shoulder_Y',
                                   46: 'left_elbow_Y', 47: 'right_elbow_Y', 48: 'left_wrist_Y', 49: 'right_wrist_Y',
                                   50: 'left_pinky_Y', 51: 'right_pinky_Y', 52: 'left_index_Y', 53: 'right_index_Y',
                                   54: 'left_thumb_Y', 55: 'right_thumb_Y', 56: 'left_hip_Y',
                                   57: 'right_hip_Y', 58: 'left_knee_Y', 59: 'right_knee_Y', 60: 'left_ankle_Y',
                                   61: 'right_ankle_Y', 62: 'left_heel_Y', 63: 'right_heel_Y', 64: 'left_foot_index_Y',
                                   65: 'right_foot_index_Y',
                                   66: 'nose_Z', 67: 'left_eye_inner_Z', 68: 'left_eye_Z', 69: 'left_eye_outer_Z',
                                   70: 'right_eye_inner_Z', 71: 'right_eye_Z', 72: 'right_eye_outer_Z',
                                   73: 'left_ear_Z', 74: 'right_ear_inner_Z', 75: 'mouth_left_Z', 76: 'mouth_right_Z',
                                   77: 'left_shoulder_Z', 78: 'right_shoulder_Z',
                                   79: 'left_elbow_Z', 80: 'right_elbow_Z', 81: 'left_wrist_Z', 82: 'right_wrist_Z',
                                   83: 'left_pinky_Z', 84: 'right_pinky_Z', 85: 'left_index_Z', 86: 'right_index_Z',
                                   87: 'left_thumb_Z', 88: 'right_thumb_Z', 89: 'left_hip_Z',
                                   90: 'right_hip_Z', 91: 'left_knee_Z', 92: 'right_knee_Z', 93: 'left_ankle_Z',
                                   94: 'right_ankle_Z', 95: 'left_heel_Z', 96: 'right_heel_Z', 97: 'left_foot_index_Z',
                                   98: 'right_foot_index_Z',
                                   99: 'nose_vis', 100: 'left_eye_inner_vis', 101: 'left_eye_vis',
                                   102: 'left_eye_outer_vis', 103: 'right_eye_inner_vis', 104: 'right_eye_vis',
                                   105: 'right_eye_outer_vis', 106: 'left_ear_vis', 107: 'right_ear_inner_vis',
                                   108: 'mouth_left_vis', 109: 'mouth_right_vis', 110: 'left_shoulder_vis',
                                   111: 'right_shoulder_vis',
                                   112: 'left_elbow_vis', 113: 'right_elbow_vis', 114: 'left_wrist_vis',
                                   115: 'right_wrist_vis', 116: 'left_pinky_vis', 117: 'right_pinky_vis',
                                   118: 'left_index_vis', 119: 'right_index_vis', 120: 'left_thumb_vis',
                                   121: 'right_thumb_vis', 122: 'left_hip_vis',
                                   123: 'right_hip_vis', 124: 'left_knee_vis', 125: 'right_knee_vis',
                                   126: 'left_ankle_vis', 127: 'right_ankle_vis', 128: 'left_heel_vis',
                                   129: 'right_heel_vis', 130: 'left_foot_index_vis', 131: 'right_foot_index_vis',
                                   132: 'width_X', 133: 'height_Y', 134: 'frame_num'
                                   }, inplace=True)
            else:
                data_temp = np.zeros((1, 132), dtype=int)
                data = np.concatenate((data_temp, [[width]], [[height]], [[frame_num]]), axis=1)
                df = pd.DataFrame(data)
                df.rename(columns={0: 'nose_X', 1: 'left_eye_inner_X', 2: 'left_eye_X', 3: 'left_eye_outer_X',
                                   4: 'right_eye_inner_X', 5: 'right_eye_X', 6: 'right_eye_outer_X', 7: 'left_ear_X',
                                   8: 'right_ear_inner_X', 9: 'mouth_left_X', 10: 'mouth_right_X',
                                   11: 'left_shoulder_X', 12: 'right_shoulder_X',
                                   13: 'left_elbow_X', 14: 'right_elbow_X', 15: 'left_wrist_X', 16: 'right_wrist_X',
                                   17: 'left_pinky_X', 18: 'right_pinky_X', 19: 'left_index_X', 20: 'right_index_X',
                                   21: 'left_thumb_X', 22: 'right_thumb_X', 23: 'left_hip_X',
                                   24: 'right_hip_X', 25: 'left_knee_X', 26: 'right_knee_X', 27: 'left_ankle_X',
                                   28: 'right_ankle_X', 29: 'left_heel_X', 30: 'right_heel_X', 31: 'left_foot_index_X',
                                   32: 'right_foot_index_X',
                                   33: 'nose_Y', 34: 'left_eye_inner_Y', 35: 'left_eye_Y', 36: 'left_eye_outer_Y',
                                   37: 'right_eye_inner_Y', 38: 'right_eye_Y', 39: 'right_eye_outer_Y',
                                   40: 'left_ear_Y', 41: 'right_ear_inner_Y', 42: 'mouth_left_Y', 43: 'mouth_right_Y',
                                   44: 'left_shoulder_Y', 45: 'right_shoulder_Y',
                                   46: 'left_elbow_Y', 47: 'right_elbow_Y', 48: 'left_wrist_Y', 49: 'right_wrist_Y',
                                   50: 'left_pinky_Y', 51: 'right_pinky_Y', 52: 'left_index_Y', 53: 'right_index_Y',
                                   54: 'left_thumb_Y', 55: 'right_thumb_Y', 56: 'left_hip_Y',
                                   57: 'right_hip_Y', 58: 'left_knee_Y', 59: 'right_knee_Y', 60: 'left_ankle_Y',
                                   61: 'right_ankle_Y', 62: 'left_heel_Y', 63: 'right_heel_Y', 64: 'left_foot_index_Y',
                                   65: 'right_foot_index_Y',
                                   66: 'nose_Z', 67: 'left_eye_inner_Z', 68: 'left_eye_Z', 69: 'left_eye_outer_Z',
                                   70: 'right_eye_inner_Z', 71: 'right_eye_Z', 72: 'right_eye_outer_Z',
                                   73: 'left_ear_Z', 74: 'right_ear_inner_Z', 75: 'mouth_left_Z', 76: 'mouth_right_Z',
                                   77: 'left_shoulder_Z', 78: 'right_shoulder_Z',
                                   79: 'left_elbow_Z', 80: 'right_elbow_Z', 81: 'left_wrist_Z', 82: 'right_wrist_Z',
                                   83: 'left_pinky_Z', 84: 'right_pinky_Z', 85: 'left_index_Z', 86: 'right_index_Z',
                                   87: 'left_thumb_Z', 88: 'right_thumb_Z', 89: 'left_hip_Z',
                                   90: 'right_hip_Z', 91: 'left_knee_Z', 92: 'right_knee_Z', 93: 'left_ankle_Z',
                                   94: 'right_ankle_Z', 95: 'left_heel_Z', 96: 'right_heel_Z', 97: 'left_foot_index_Z',
                                   98: 'right_foot_index_Z',
                                   99: 'nose_vis', 100: 'left_eye_inner_vis', 101: 'left_eye_vis',
                                   102: 'left_eye_outer_vis', 103: 'right_eye_inner_vis', 104: 'right_eye_vis',
                                   105: 'right_eye_outer_vis', 106: 'left_ear_vis', 107: 'right_ear_inner_vis',
                                   108: 'mouth_left_vis', 109: 'mouth_right_vis', 110: 'left_shoulder_vis',
                                   111: 'right_shoulder_vis',
                                   112: 'left_elbow_vis', 113: 'right_elbow_vis', 114: 'left_wrist_vis',
                                   115: 'right_wrist_vis', 116: 'left_pinky_vis', 117: 'right_pinky_vis',
                                   118: 'left_index_vis', 119: 'right_index_vis', 120: 'left_thumb_vis',
                                   121: 'right_thumb_vis', 122: 'left_hip_vis',
                                   123: 'right_hip_vis', 124: 'left_knee_vis', 125: 'right_knee_vis',
                                   126: 'left_ankle_vis', 127: 'right_ankle_vis', 128: 'left_heel_vis',
                                   129: 'right_heel_vis', 130: 'left_foot_index_vis', 131: 'right_foot_index_vis',
                                   132: 'width_X', 133: 'height_Y', 134: 'frame_num'
                                   }, inplace=True)

            # This one to export 1 file per row

            df.to_csv((dirDAT + file_name + '_Pose_Frame ' + str(int(frame_num)) + '.csv'),
                      mode='w', sep=',', index=False, header=True)

        # Create the black background
        image_out = image_shape_zero
        image_out = image_out / image_out.max()  # normalizes data in range 0 - 255
        image_out = 255 * image_out
        img_out = image_out.astype(np.uint8)

        if export_vid > 1:
            # write the frame and export the mp4 file with a black background
            out_black.write(img_out)
            # write the frame and export the mp4 file with the original video as background
            out.write(image)

        # remove if you don't want to see the video
        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if export_data > 1 > one_file_per_frame:
            # read the exported data and export one file and replace the headers
            temp = pd.read_csv((dirDAT + file_name + '_Pose_out_dat.csv'), sep=',', header=None)
            temp.rename(columns={0: 'nose_X', 1: 'left_eye_inner_X', 2: 'left_eye_X', 3: 'left_eye_outer_X',
                                 4: 'right_eye_inner_X', 5: 'right_eye_X', 6: 'right_eye_outer_X', 7: 'left_ear_X',
                                 8: 'right_ear_inner_X', 9: 'mouth_left_X', 10: 'mouth_right_X', 11: 'left_shoulder_X',
                                 12: 'right_shoulder_X',
                                 13: 'left_elbow_X', 14: 'right_elbow_X', 15: 'left_wrist_X', 16: 'right_wrist_X',
                                 17: 'left_pinky_X', 18: 'right_pinky_X', 19: 'left_index_X', 20: 'right_index_X',
                                 21: 'left_thumb_X', 22: 'right_thumb_X', 23: 'left_hip_X',
                                 24: 'right_hip_X', 25: 'left_knee_X', 26: 'right_knee_X', 27: 'left_ankle_X',
                                 28: 'right_ankle_X', 29: 'left_heel_X', 30: 'right_heel_X', 31: 'left_foot_index_X',
                                 32: 'right_foot_index_X',
                                 33: 'nose_Y', 34: 'left_eye_inner_Y', 35: 'left_eye_Y', 36: 'left_eye_outer_Y',
                                 37: 'right_eye_inner_Y', 38: 'right_eye_Y', 39: 'right_eye_outer_Y', 40: 'left_ear_Y',
                                 41: 'right_ear_inner_Y', 42: 'mouth_left_Y', 43: 'mouth_right_Y',
                                 44: 'left_shoulder_Y', 45: 'right_shoulder_Y',
                                 46: 'left_elbow_Y', 47: 'right_elbow_Y', 48: 'left_wrist_Y', 49: 'right_wrist_Y',
                                 50: 'left_pinky_Y', 51: 'right_pinky_Y', 52: 'left_index_Y', 53: 'right_index_Y',
                                 54: 'left_thumb_Y', 55: 'right_thumb_Y', 56: 'left_hip_Y',
                                 57: 'right_hip_Y', 58: 'left_knee_Y', 59: 'right_knee_Y', 60: 'left_ankle_Y',
                                 61: 'right_ankle_Y', 62: 'left_heel_Y', 63: 'right_heel_Y', 64: 'left_foot_index_Y',
                                 65: 'right_foot_index_Y',
                                 66: 'nose_Z', 67: 'left_eye_inner_Z', 68: 'left_eye_Z', 69: 'left_eye_outer_Z',
                                 70: 'right_eye_inner_Z', 71: 'right_eye_Z', 72: 'right_eye_outer_Z', 73: 'left_ear_Z',
                                 74: 'right_ear_inner_Z', 75: 'mouth_left_Z', 76: 'mouth_right_Z',
                                 77: 'left_shoulder_Z', 78: 'right_shoulder_Z',
                                 79: 'left_elbow_Z', 80: 'right_elbow_Z', 81: 'left_wrist_Z', 82: 'right_wrist_Z',
                                 83: 'left_pinky_Z', 84: 'right_pinky_Z', 85: 'left_index_Z', 86: 'right_index_Z',
                                 87: 'left_thumb_Z', 88: 'right_thumb_Z', 89: 'left_hip_Z',
                                 90: 'right_hip_Z', 91: 'left_knee_Z', 92: 'right_knee_Z', 93: 'left_ankle_Z',
                                 94: 'right_ankle_Z', 95: 'left_heel_Z', 96: 'right_heel_Z', 97: 'left_foot_index_Z',
                                 98: 'right_foot_index_Z',
                                 99: 'nose_vis', 100: 'left_eye_inner_vis', 101: 'left_eye_vis',
                                 102: 'left_eye_outer_vis', 103: 'right_eye_inner_vis', 104: 'right_eye_vis',
                                 105: 'right_eye_outer_vis', 106: 'left_ear_vis', 107: 'right_ear_inner_vis',
                                 108: 'mouth_left_vis', 109: 'mouth_right_vis', 110: 'left_shoulder_vis',
                                 111: 'right_shoulder_vis',
                                 112: 'left_elbow_vis', 113: 'right_elbow_vis', 114: 'left_wrist_vis',
                                 115: 'right_wrist_vis', 116: 'left_pinky_vis', 117: 'right_pinky_vis',
                                 118: 'left_index_vis', 119: 'right_index_vis', 120: 'left_thumb_vis',
                                 121: 'right_thumb_vis', 122: 'left_hip_vis',
                                 123: 'right_hip_vis', 124: 'left_knee_vis', 125: 'right_knee_vis',
                                 126: 'left_ankle_vis', 127: 'right_ankle_vis', 128: 'left_heel_vis',
                                 129: 'right_heel_vis', 130: 'left_foot_index_vis', 131: 'right_foot_index_vis',
                                 132: 'width_X', 133: 'height_Y', 134: 'frame_num'
                                 }, inplace=True)
            temp.to_csv((dirDAT + file_name + '_Pose_out_dat.csv'), mode='w', sep=',', index=True, header=True)
            temp.to_json((dirDAT + file_name + '_Pose_out_dat.json'), index=True)

cap.release()
out.release()
out_black.release()
cv2.destroyAllWindows()


if export_data > 1 and one_file_per_frame > 1:
    # Read all csv in this folder
    os.chdir('/Users/juliantanquahjian/mediapipe_working/videos/export_dat/' + file_name)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    # combine all files in the list
    temp = pd.concat([pd.read_csv(f) for f in all_filenames], axis=0)
    temp = temp.sort_values(by=['frame_num'], axis=0, ascending=True)

    if not os.path.exists("combined/"):
        os.makedirs("combined/")

    temp.reset_index(inplace=True)
    temp.to_csv(("combined/" + file_name + '_Pose_out_dat.csv'), mode='w', sep=',', index=True, header=True)
    temp.to_json(("combined/" + file_name + '_Pose_out_dat.json'), index=True)
