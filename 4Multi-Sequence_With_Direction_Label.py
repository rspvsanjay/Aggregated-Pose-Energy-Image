import cv2
import os
import numpy as np
import re  # For extracting frame indices from filenames

def compute_optical_flow(prev_image, curr_image):
    """Compute optical flow between two consecutive images."""
    flow = cv2.calcOpticalFlowFarneback(prev_image, curr_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def detect_direction(prev_image, curr_image):
    """Determine movement direction based on optical flow."""
    flow = compute_optical_flow(prev_image, curr_image)
    flow_x = flow[..., 0]  # Horizontal flow
    horizontal_flow = np.sum(flow_x)
    return "left_to_right" if horizontal_flow > 0 else "right_to_left"

def extract_frame_index(filename):
    """Extract the frame index from the filename."""
    match = re.search(r'frame(\d+)', filename)
    return int(match.group(1)) if match else None

def remove_extension_from_filename(filename):
    """Remove the .png, .jpg, or .jpeg extension from the filename."""
    return re.sub(r'\.(png|jpg|jpeg)', '', filename, flags=re.IGNORECASE)

def process_images(image_dir, save_base_dir):
    """Processes images, detects direction changes, and saves all images in separate folders until a direction change."""
    os.makedirs(save_base_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if len(image_files) < 2:
        print(f"Not enough images in {image_dir} to process.")
        return
    prev_image = cv2.imread(os.path.join(image_dir, image_files[0]), cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3, 3), np.uint8)
    prev_direction = detect_direction(prev_image, prev_image)
    sequence_counter = 1
    sequence_counter_str = f"{sequence_counter:02}"
    save_dir = os.path.join(save_base_dir, f"Sequence_{sequence_counter_str}_{prev_direction}")
    os.makedirs(save_dir, exist_ok=True)
    first_filename = remove_extension_from_filename(image_files[0])
    cv2.imwrite(os.path.join(save_dir, f"{first_filename}_{prev_direction}.png"), prev_image)
    for i in range(1, len(image_files)):
        curr_image = cv2.imread(os.path.join(image_dir, image_files[i]), cv2.IMREAD_GRAYSCALE)
        curr_direction = detect_direction(prev_image, curr_image)
        prev_index = extract_frame_index(image_files[i - 1])
        curr_index = extract_frame_index(image_files[i])
        if curr_direction != prev_direction or abs(curr_index - prev_index) > 15:
            sequence_counter += 1
            sequence_counter_str = f"{sequence_counter:02}"
            save_dir = os.path.join(save_base_dir, f"Sequence_{sequence_counter_str}_{curr_direction}")
            os.makedirs(save_dir, exist_ok=True)
        curr_filename = remove_extension_from_filename(image_files[i])
        save_path = os.path.join(save_dir, f"{curr_filename}_{curr_direction}.png")
        cv2.imwrite(save_path, curr_image)
        prev_image = curr_image
        prev_direction = curr_direction
        print(f"Saved {image_files[i]} in {save_dir}, Direction: {curr_direction}, Index Gap: {abs(curr_index - prev_index)}")

def process_for_multiple_subjects(base_image_dir, base_save_dir, start_subject=1, end_subject=130):
    """Processes images for multiple subjects and saves direction sequences in separate folders."""
    for subject in range(start_subject, end_subject + 1):
        subject_id = f"{subject:03d}"
        subject_image_dir = os.path.join(base_image_dir, str(subject_id))
        subject_save_dir = os.path.join(base_save_dir, str(subject_id))
        if os.path.exists(subject_image_dir):
            print(f"Processing subject {subject_id}...")
            process_images(subject_image_dir, subject_save_dir)
        else:
            print(f"Directory for subject {subject_id} does not exist.")

# Define base directories
image_base_directory = f'E:\\Gait_IIT_BHU_Data\\Filtered_Images\\'
save_base_directory = f'E:\\Gait_IIT_BHU_Data\\Sequences_with_Labeled_Direction\\'
os.makedirs(save_base_directory, exist_ok=True)

# Run the function to process images for subjects 201 to 220
process_for_multiple_subjects(image_base_directory, save_base_directory)