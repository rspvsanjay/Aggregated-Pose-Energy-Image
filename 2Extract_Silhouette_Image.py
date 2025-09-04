import cv2
import numpy as np
import os
import gc  # For garbage collection


def extract_silhouette(background, frame, threshold_value, kernel_size):
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(background_gray, frame_gray)
    _, silhouette = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, kernel)
    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, kernel)

    return silhouette


def process_videos_in_subjects(root_video_folder, root_background_folder, output_root_folder):
    for subject_id in range(1, 134):
        subject_id1 = f"{subject_id:03d}"
        subject_folder = os.path.join(root_video_folder, str(subject_id1))

        if not os.path.exists(subject_folder):
            print(f"Subject {subject_id1} folder not found.")
            continue

        files_in_subject_folder = os.listdir(subject_folder)
        print(f"Files in subject {subject_id1} folder: {files_in_subject_folder}")

        video_files = [f for f in files_in_subject_folder if f.lower().endswith(".mp4")]
        if not video_files:
            print(f"No video files found for subject {subject_id1}.")
            continue

        video_path = os.path.join(subject_folder, video_files[0])

        background_folder_path = os.path.join(root_background_folder, str(subject_id1))
        background_path = os.path.join(background_folder_path, "background.png")

        if not os.path.exists(background_path):
            print(f"Background image for subject {subject_id1} not found at {background_path}.")
            continue

        background = cv2.imread(background_path)
        if background is None:
            print(f"Error loading background for subject {subject_id1}.")
            continue

        print(f"Processing video {video_path} for subject {subject_id1}...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video {video_path}.")
            continue

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for threshold_value in range(56, 57, 2):
                for kernel_size in range(4, 5, 1):
                    silhouette = extract_silhouette(background, frame, threshold_value, kernel_size)

                    th_k_folder = os.path.join(output_root_folder, f"data_th_{threshold_value}_k_{kernel_size}")
                    subject_output_folder = os.path.join(th_k_folder, str(subject_id1))
                    os.makedirs(subject_output_folder, exist_ok=True)

                    output_filename = f"silhouette_frame{frame_count:04d}_th{threshold_value}_k{kernel_size}.png"
                    output_path = os.path.join(subject_output_folder, output_filename)
                    cv2.imwrite(output_path, silhouette)

                    saved_count += 1

                    # Free memory of silhouette image
                    del silhouette

            frame_count += 1

            # Free memory of the current frame
            del frame

        cap.release()
        del cap
        del background
        gc.collect()  # Force garbage collection

        print(f"Processed {frame_count} frames for subject {subject_id1}. {saved_count} silhouettes saved.\n")

    print("Finished processing all subjects.")


# Example Usage
root_video_folder = r"E:\Gait_IIT_BHU_Data\Gait_IIT_BHU_Videos"
root_background_folder = r"E:\Gait_IIT_BHU_Data\Gait_IIT_BHU_background"
output_root_folder = r"E:\Gait_IIT_BHU_Data\Gait_IIT_BHU_Silhouette"

process_videos_in_subjects(root_video_folder, root_background_folder, output_root_folder)