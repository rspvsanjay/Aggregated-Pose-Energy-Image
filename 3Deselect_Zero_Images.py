import cv2
import os
import numpy as np

start_sub = 1
end_sub = 133

def plot_sum_pixel_values(image_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print(f"No images found in {image_dir}.")
        return

    sum_pixel_values = []
    saved_pixel_values = []

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            sum_pixel_value = np.sum(image)
            sum_pixel_values.append((image_file, sum_pixel_value, image))
        else:
            print(f"Could not load image: {image_file}")

    if not sum_pixel_values:
        print("No valid images processed.")
        return

    pixel_sums = [val[1] for val in sum_pixel_values]
    avg_value = np.mean(pixel_sums)

    print(f"Processing {image_dir}:")
    print(f"Average Sum of Pixel Values: {avg_value:.2f}")

    save_count = 0
    for image_file, sum_pixel_value, image in sum_pixel_values:
        if sum_pixel_value > (avg_value * 1.1):
            save_path = os.path.join(save_dir, image_file)
            cv2.imwrite(save_path, image)
            saved_pixel_values.append(sum_pixel_value)
            save_count += 1

    print(f"Saved {save_count} images to '{save_dir}'.")

def retrieve_missing_frames(filtered_dir, original_dir, save_dir, th, k, threshold_diff=15):
    os.makedirs(save_dir, exist_ok=True)
    filtered_images = sorted([f for f in os.listdir(filtered_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    filtered_indices = []
    for f in filtered_images:
        try:
            frame_index = int(f.split('_')[1].replace('frame', ''))
            filtered_indices.append(frame_index)
        except ValueError:
            print(f"Skipping invalid filename: {f}")

    filtered_indices.sort()

    for i in range(len(filtered_indices) - 1):
        current_index = filtered_indices[i]
        next_index = filtered_indices[i + 1]

        if next_index - current_index > 1 and next_index - current_index < threshold_diff:
            print(f"Retrieving missing frames for subject in {original_dir}: {current_index} to {next_index}")

            for missing_index in range(current_index + 1, next_index):
                missing_frame_name = f"silhouette_frame{missing_index:04d}_th{th}_k{k}.png"
                missing_frame_path = os.path.join(original_dir, missing_frame_name)
                save_path = os.path.join(save_dir, missing_frame_name)
                if os.path.exists(missing_frame_path):
                    image = cv2.imread(missing_frame_path, cv2.IMREAD_GRAYSCALE)
                    cv2.imwrite(save_path, image)
                    print(f"Saved missing frame: {missing_frame_name}")
                else:
                    print(f"Frame {missing_frame_name} not found in original directory.")

base_input_dir = r"E:\Gait_IIT_BHU_Data\Gait_IIT_BHU_Silhouette\data_th_56_k_4"
base_filtered_dir = r"E:\\Gait_IIT_BHU_Data\Filtered_Images"
th=56
k=4
os.makedirs(base_filtered_dir, exist_ok=True)
for subject_id in range(start_sub, end_sub + 1):
    subject_id1 = f"{subject_id:03d}"
    subject_dir = os.path.join(base_input_dir, str(subject_id1))
    subject_filtered_dir = os.path.join(base_filtered_dir, str(subject_id1))

    if os.path.exists(subject_dir):
        print(f"Processing Subject: {subject_id1}")
        plot_sum_pixel_values(subject_dir, subject_filtered_dir)
        retrieve_missing_frames(subject_filtered_dir, subject_dir, subject_filtered_dir, th, k)
    else:
        print(f"Subject folder not found: {subject_dir}")
