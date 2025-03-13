import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pandas as pd
import os
import cv2
import time
import glob


def create_output_directories(video_file_name):
    """
    Creates directories for exporting videos and data if they don't exist.
    """
    dirVID = f'videos/export_vid/{video_file_name}/'
    dirDAT = f'videos/export_dat/{video_file_name}/'

    os.makedirs(dirVID, exist_ok=True)
    os.makedirs(dirDAT, exist_ok=True)

    return dirVID, dirDAT


def clean_previous_data(file_path):
    """
    Deletes previous data file if it exists.
    """
    if os.path.exists(file_path):
        os.remove(file_path)


# create the function to draw the skeleton on the video
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(annotated_image, pose_landmarks_proto, solutions.pose.POSE_CONNECTIONS,
                                               solutions.drawing_styles.get_default_pose_landmarks_style())

    return annotated_image


def add_frame_num_text(image, frame_num, position=(50, 50), text_color=(0, 0, 255), background_color=(255, 255, 255)):
    frame_number_text = "Frame: " + str(int(frame_num))

    # Determine the text size
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 1
    text_thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(frame_number_text, text_font, text_scale, text_thickness)

    # Add padding around the text
    background_padding_x = 10  # Adjust the padding around the text horizontally as needed
    background_padding_y = 5   # Adjust the padding around the text vertically as needed

    # Calculate background size based on text size and padding
    background_width = text_width + 2 * background_padding_x
    background_height = text_height + 2 * background_padding_y

    # Add a white background behind the text
    background_start = (position[0], position[1])
    background_end = (background_start[0] + background_width, background_start[1] + background_height)
    cv2.rectangle(image, background_start, background_end, background_color, -1)

    # Add the frame number text
    text_position = (position[0] + background_padding_x, position[1] + text_height + background_padding_y)
    cv2.putText(image, frame_number_text, text_position, text_font, text_scale, text_color, text_thickness, cv2.LINE_AA)

    return image


# function to get data
def get_data(st_time, video_file_name, video_file, dirVID):

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter((dirVID + video_file_name + '_Pose_out.mp4'),
                          fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a pose landmarker instance with the video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3)

    with PoseLandmarker.create_from_options(options) as landmarker:

        # Create a temp dataframe to store the data before exporting into csv
        # temp_pose_df1 = pd.DataFrame()
        # temp_pose_df2 = pd.DataFrame()

        temp_pose_df1_list = []
        temp_pose_df2_list = []

        while cap.isOpened():
            end_time_frame = time.time()
            time_taken_frame = end_time_frame - st_time
            fps_frame = round((1 / time_taken_frame), 2)
            st_time = end_time_frame
            print("FPS: ", fps_frame)

            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # print('Processed Frame ', int(frame_num))
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")

                # If loading a video, use 'break' instead of 'continue'.
                break

            image = add_frame_num_text(image, frame_num)

            # If you want to mask the video = play with X and Y
            # change the values if you want pose_landmark to output normalised or not (to video resolution size)
            NORMALISE = 1
            MASK = 1  # default value == 1 --> do not change, adjust to 0 for all below if you want no mask
            left = 0  # how much to remove from left side (in %)
            right = 0  # how much to remove from right side (in %)
            top = 0  # how much to remove from top (in %)
            bottom = 0  # how much to remove from bottom (in %)

            if MASK == 1:
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                masks = np.zeros([int(height), int(width)], dtype='uint8')
                top_left_corner = (int(left / 100 * width), int(top / 100 * height))
                bottom_right_corner = (int(width - right / 100 * width), int(height - bottom / 100 * height))
                masks = cv2.rectangle(masks, top_left_corner, bottom_right_corner, (255, 255, 255), -1)

            if MASK == 1 and left or right or top or bottom > 0:  # if MASK YES
                timestamp = 1 / fps * frame_num
                image_cut = cv2.bitwise_and(image, image, mask=masks)
                mp_image_cut = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_cut)
                detection_result = landmarker.detect_for_video(mp_image_cut, int(timestamp * 1000))

            else:
                timestamp = 1 / fps * frame_num
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                detection_result = landmarker.detect_for_video(mp_image, int(timestamp * 1000))

            # unpack list from the results
            flat_list = [item for sublist in detection_result.pose_landmarks for item in sublist]
            flat_list_world = [item for sublist in detection_result.pose_world_landmarks for item in sublist]

            # transform normalized and world results into a dataframe
            pose_landmarks = pd.DataFrame(flat_list)

            # manually input data as 0.0 if human not detected
            if pose_landmarks.empty:
                num_elements = 33  # <-- number of pose landmarks detected
                zero_list = [0.0] * num_elements
                zero_cols = ["x", "y", "z", "visibility", "presence"]
                df_zeros = {col: zero_list.copy() for col in zero_cols}
                pose_landmarks = pd.DataFrame(df_zeros)

            if NORMALISE == 1:
                pose_landmarks["x"] = pose_landmarks["x"] * width
                pose_landmarks["y"] = pose_landmarks["y"] * height
                pose_landmarks["z"] = pose_landmarks["z"] * width
            else:
                pose_landmarks["x"] = pose_landmarks["x"]
                pose_landmarks["y"] = pose_landmarks["y"]
                pose_landmarks["z"] = pose_landmarks["z"]

            pose_world_landmarks = pd.DataFrame(flat_list_world)

            # manually input data as 0.0 if human not detected
            if pose_world_landmarks.empty:
                num_elements = 33  # <-- number of pose landmarks detected
                zero_list = [0.0] * num_elements
                zero_cols = ["x", "y", "z", "visibility", "presence"]
                df_zeros = {col: zero_list.copy() for col in zero_cols}

                pose_world_landmarks = pd.DataFrame(df_zeros)

            # get the label of the point in the output data
            columns = pd.DataFrame({'body_parts': [
                '00_nose', '01_left_eye_(inner)', '02_left_eye',
                '03_left_eye_(outer)', '04_right_eye_(inner)', '05_right_eye',
                '06_right_eye_(outer)', '07_left_ear', '08_right_ear',
                '09_mouth_(left)', '10_mouth_(right)', '11_left_shoulder',
                '12_right_shoulder', '13_left_elbow', '14_right_elbow',
                '15_left_wrist', '16_right_wrist', '17_left_pinky',
                '18_right_pinky', '19_left_index', '20_right_index',
                '21_left_thumb', '22_right_thumb', '23_left_hip',
                '24_right_hip', '25_left_knee', '26_right_knee',
                '27_left_ankle', '28_right_ankle', '29_left_heel',
                '30_right_heel', '31_left_foot_index', '32_right_foot_index']})

            # get the frame num in the output data as well
            frame_list = pd.DataFrame({'frame': [
                frame_num, frame_num, frame_num,
                frame_num, frame_num, frame_num,
                frame_num, frame_num, frame_num,
                frame_num, frame_num, frame_num,
                frame_num, frame_num, frame_num,
                frame_num, frame_num, frame_num,
                frame_num, frame_num, frame_num,
                frame_num, frame_num, frame_num,
                frame_num, frame_num, frame_num,
                frame_num, frame_num, frame_num,
                frame_num, frame_num, frame_num]})

            # combine results into a dataframe
            df1 = pd.concat([columns, pose_landmarks, frame_list], axis=1)
            df2 = pd.concat([columns, pose_world_landmarks, frame_list], axis=1)

            # Flatten the table for df1
            pivot_df1 = pd.pivot_table(df1, values=['x', 'y', 'z', 'visibility', 'presence'],
                                       index='frame', columns='body_parts', aggfunc='first')

            # Resetting the index to make 'frame' a column again
            pivot_df1.reset_index(inplace=True)

            # Flattening the MultiIndex columns into a single-level column
            pivot_df1.columns = [f'{col[1]}_{col[0]}' if col[1] else col[0] for col in pivot_df1.columns]

            # Flatten the table for df2
            pivot_df2 = pd.pivot_table(df2, values=['x', 'y', 'z', 'visibility', 'presence'],
                                       index='frame', columns='body_parts', aggfunc='first')

            # Resetting the index to make 'frame' a column again
            pivot_df2.reset_index(inplace=True)

            # Flattening the MultiIndex columns into a single-level column
            pivot_df2.columns = [f'{col[1]}_{col[0]}' if col[1] else col[0] for col in pivot_df2.columns]

            # Re-ordering the variable column to (x, y, z, visibility, presence)
            desired_columns_order = \
                ['frame', '00_nose_x', '01_left_eye_(inner)_x',
                 '02_left_eye_x', '03_left_eye_(outer)_x',
                 '04_right_eye_(inner)_x', '05_right_eye_x',
                 '06_right_eye_(outer)_x', '07_left_ear_x',
                 '08_right_ear_x', '09_mouth_(left)_x', '10_mouth_(right)_x',
                 '11_left_shoulder_x', '12_right_shoulder_x', '13_left_elbow_x',
                 '14_right_elbow_x', '15_left_wrist_x', '16_right_wrist_x',
                 '17_left_pinky_x', '18_right_pinky_x', '19_left_index_x',
                 '20_right_index_x', '21_left_thumb_x', '22_right_thumb_x',
                 '23_left_hip_x', '24_right_hip_x', '25_left_knee_x',
                 '26_right_knee_x', '27_left_ankle_x', '28_right_ankle_x',
                 '29_left_heel_x', '30_right_heel_x', '31_left_foot_index_x',
                 '32_right_foot_index_x', '00_nose_y', '01_left_eye_(inner)_y',
                 '02_left_eye_y', '03_left_eye_(outer)_y',
                 '04_right_eye_(inner)_y', '05_right_eye_y',
                 '06_right_eye_(outer)_y', '07_left_ear_y',
                 '08_right_ear_y', '09_mouth_(left)_y', '10_mouth_(right)_y',
                 '11_left_shoulder_y', '12_right_shoulder_y', '13_left_elbow_y',
                 '14_right_elbow_y', '15_left_wrist_y', '16_right_wrist_y',
                 '17_left_pinky_y', '18_right_pinky_y', '19_left_index_y',
                 '20_right_index_y', '21_left_thumb_y', '22_right_thumb_y',
                 '23_left_hip_y', '24_right_hip_y', '25_left_knee_y',
                 '26_right_knee_y', '27_left_ankle_y', '28_right_ankle_y',
                 '29_left_heel_y', '30_right_heel_y', '31_left_foot_index_y',
                 '32_right_foot_index_y', '00_nose_z', '01_left_eye_(inner)_z',
                 '02_left_eye_z', '03_left_eye_(outer)_z',
                 '04_right_eye_(inner)_z', '05_right_eye_z',
                 '06_right_eye_(outer)_z', '07_left_ear_z',
                 '08_right_ear_z', '09_mouth_(left)_z', '10_mouth_(right)_z',
                 '11_left_shoulder_z', '12_right_shoulder_z', '13_left_elbow_z',
                 '14_right_elbow_z', '15_left_wrist_z', '16_right_wrist_z',
                 '17_left_pinky_z', '18_right_pinky_z', '19_left_index_z',
                 '20_right_index_z', '21_left_thumb_z', '22_right_thumb_z',
                 '23_left_hip_z', '24_right_hip_z', '25_left_knee_z',
                 '26_right_knee_z', '27_left_ankle_z', '28_right_ankle_z',
                 '29_left_heel_z', '30_right_heel_z', '31_left_foot_index_z',
                 '32_right_foot_index_z', '00_nose_visibility', '01_left_eye_(inner)_visibility',
                 '02_left_eye_visibility', '03_left_eye_(outer)_visibility',
                 '04_right_eye_(inner)_visibility', '05_right_eye_visibility',
                 '06_right_eye_(outer)_visibility', '07_left_ear_visibility',
                 '08_right_ear_visibility', '09_mouth_(left)_visibility', '10_mouth_(right)_visibility',
                 '11_left_shoulder_visibility', '12_right_shoulder_visibility', '13_left_elbow_visibility',
                 '14_right_elbow_visibility', '15_left_wrist_visibility', '16_right_wrist_visibility',
                 '17_left_pinky_visibility', '18_right_pinky_visibility', '19_left_index_visibility',
                 '20_right_index_visibility', '21_left_thumb_visibility', '22_right_thumb_visibility',
                 '23_left_hip_visibility', '24_right_hip_visibility', '25_left_knee_visibility',
                 '26_right_knee_visibility', '27_left_ankle_visibility', '28_right_ankle_visibility',
                 '29_left_heel_visibility', '30_right_heel_visibility', '31_left_foot_index_visibility',
                 '32_right_foot_index_visibility', '00_nose_presence', '01_left_eye_(inner)_presence',
                 '02_left_eye_presence', '03_left_eye_(outer)_presence',
                 '04_right_eye_(inner)_presence', '05_right_eye_presence',
                 '06_right_eye_(outer)_presence', '07_left_ear_presence',
                 '08_right_ear_presence', '09_mouth_(left)_presence', '10_mouth_(right)_presence',
                 '11_left_shoulder_presence', '12_right_shoulder_presence', '13_left_elbow_presence',
                 '14_right_elbow_presence', '15_left_wrist_presence', '16_right_wrist_presence',
                 '17_left_pinky_presence', '18_right_pinky_presence', '19_left_index_presence',
                 '20_right_index_presence', '21_left_thumb_presence', '22_right_thumb_presence',
                 '23_left_hip_presence', '24_right_hip_presence', '25_left_knee_presence',
                 '26_right_knee_presence', '27_left_ankle_presence', '28_right_ankle_presence',
                 '29_left_heel_presence', '30_right_heel_presence', '31_left_foot_index_presence',
                 '32_right_foot_index_presence']

            # Replace columns
            pivot_df1 = pivot_df1[desired_columns_order]
            pivot_df2 = pivot_df2[desired_columns_order]

            temp_pose_df1_list.append(pivot_df1)
            temp_pose_df2_list.append(pivot_df2)

            # export tracking on original
            annotated_image = draw_landmarks_on_image(image, detection_result)
            cv2.imshow('MediaPipe Annotated', annotated_image)
            cv2.waitKey(1)
            out.write(annotated_image)

            # export tracking on black video
            # annotated_image_black = draw_landmarks_on_image(img_out, detection_result)
            # out_black.write(annotated_image_black)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    return temp_pose_df1_list, temp_pose_df2_list


def save_data(video_file_name, dirDAT, temp_pose_df1_list, temp_pose_df2_list, chunk_size=500):
    temp_pose_df1 = pd.concat(temp_pose_df1_list)
    temp_pose_df2 = pd.concat(temp_pose_df2_list)

    # Export df to csv
    temp_pose_df1.to_csv((dirDAT + video_file_name + '_Pose_combined' + '.csv'),
                         mode='w', sep=',', index=False, header=True)
    temp_pose_df2.to_csv((dirDAT + video_file_name + '_Pose_world_combined' + '.csv'),
                         mode='w', sep=',', index=False, header=True)



if __name__ == "__main__":
    input_videos_folder = 'videos/'
    model_path = 'models/pose_landmarker_full.task'

    st_time = time.time()
    print("Python file running:", os.path.basename(__file__))
    print("Start Time: ", time.strftime("%H:%M:%S"))
    start_time = time.perf_counter()

    video_files = glob.glob(os.path.join(input_videos_folder, '*.mp4'))

    for video_file in video_files:
        video_file_name = os.path.splitext(os.path.basename(video_file))[0]
        dirVID, dirDAT = create_output_directories(video_file_name)

        filePath_temp = f'{dirDAT}{video_file_name}_Pose_out_dat.csv'
        clean_previous_data(filePath_temp)

        temp_pose_df1_list, temp_pose_df2_list = get_data(st_time, video_file_name, video_file, dirVID)
        save_data(video_file_name, dirDAT, temp_pose_df1_list, temp_pose_df2_list)

    end_time = time.perf_counter()
    runtime_seconds = round((end_time - start_time), 2)
    print("End Time: ", time.strftime("%H:%M:%S"))
    print(f"Runtime: {runtime_seconds} seconds")