import numpy as np
import cv2
import os
from collections import Counter


def get_background(frames):
    height, width = frames[0].shape
    background = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            pixels = [frame[y, x] for frame in frames]  # Collect all pixel values at (x, y)
            most_common_pixel = Counter(pixels).most_common(1)[0][0]  # Find the most frequent pixel
            background[y, x] = most_common_pixel

    return background


def get_first_video(subject_path):
    if not os.path.exists(subject_path):
        print(f"Subject folder not found: {subject_path}")
        return None

    files = os.listdir(subject_path)
    print(f"Files in {subject_path}: {files}")  # Debugging print

    # Convert all extensions to lowercase for filtering
    video_files = sorted([f for f in files if f.lower().endswith(".mp4")])

    if not video_files:
        print(f"No video files found in {subject_path}")
        return None

    return video_files[0]


# Example usage
if __name__ == "__main__":
    # Updated: subjects numbered from 001 to 133 (zero-padded)
    subjects = [f"{i:03d}" for i in range(1, 134)]
    root_video_folder = r"E:\Gait_IIT_BHU_Data\Gait_IIT_BHU_Videos"
    output_background_folder = r"E:\Gait_IIT_BHU_Data\Gait_IIT_BHU_background"

    for subject in subjects:
        subject_path = os.path.join(root_video_folder, subject)
        first_video = get_first_video(subject_path)

        if not first_video:
            print(f"No video found for subject {subject}")
            continue

        video_path = os.path.join(subject_path, first_video)
        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frames = []

        for _ in range(100):
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
            frames.append(gray_frame)

        cap.release()

        if frames:
            background = get_background(frames)
            subject_output_folder = os.path.join(output_background_folder, subject)
            os.makedirs(subject_output_folder, exist_ok=True)
            output_path = os.path.join(subject_output_folder, "background.png")
            cv2.imwrite(output_path, background)
            print(f"Background image saved for subject {subject} at {output_path}")