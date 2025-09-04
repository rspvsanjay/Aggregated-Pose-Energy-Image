import shutil
import cv2
import os
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import find_peaks

def find_nearest_peaks(original_signal, smoothed_peaks):
    """Finds the nearest peaks in the original signal corresponding to the smoothed peaks."""
    original_peaks, _ = find_peaks(original_signal)

    if not original_peaks.size:  # Check if original_peaks is empty
        print("Warning: No peaks found in original signal.")
        return []  # Return empty list to prevent error

    nearest_peaks = []
    for sp in smoothed_peaks:
        nearest = min(original_peaks, key=lambda x: abs(x - sp))
        nearest_peaks.append(nearest)

    return nearest_peaks


def find_nearest_peaks1(original_peaks, smoothed_peaks):
    """Finds the nearest peaks in original_peaks corresponding to smoothed_peaks."""

    if not original_peaks:
        print("Warning: No peaks found in original signal.")
        return []

    nearest_peaks = []
    for sp in smoothed_peaks:
        nearest = min(original_peaks, key=lambda x: abs(x - sp))  # Match closest peak
        nearest_peaks.append(nearest)

    return nearest_peaks

def compute_correlations(image_paths, avg_image):
    """Computes Pearson correlation between each image and the average image."""
    correlations = []
    avg_flat = avg_image.flatten()

    for path in image_paths:
        img = read_image(path).flatten()
        corr, _ = pearsonr(img, avg_flat)
        correlations.append(corr)

    return correlations


def smooth_signal(signal, window_size=5):
    """Smooths a signal using a simple moving average while preserving length."""
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")

    half_window = window_size // 2
    smoothed = np.convolve(signal, np.ones(window_size) / window_size, mode='valid')

    # Preserve the original length by copying the first few values as-is
    padded_smoothed = np.concatenate((signal[:half_window], smoothed, signal[-half_window:]))

    return padded_smoothed


def compute_average_image(image_paths):
    """Computes the average image from a list of image paths."""
    images = [read_image(p).astype(np.float32) for p in image_paths]
    avg_image = np.mean(images, axis=0)
    return avg_image.astype(np.uint8)

def get_image_paths(folder_path):
    """Returns sorted list of image file paths in the given folder."""
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    image_files.sort()  # Ensure proper ordering
    return [os.path.join(folder_path, f) for f in image_files]

def read_image(image_path, size=(256, 256)):
    """Reads and resizes an image to a fixed size."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def count_frames_per_sequence(root_folder, save_folder, avg_image):
    """Counts the number of frames in each sequence and prints the count if >= 8."""
    #for subject in os.listdir(root_folder):
    for subject in range(1, 134):
        subject_str = f"{subject:03d}"
        #subject = str(subject)  # Convert subject number to string
        subject_path = os.path.join(root_folder, subject_str)
        if os.path.isdir(subject_path):
            for sequence in os.listdir(subject_path):
                sequence_path = os.path.join(subject_path, sequence)
                if os.path.isdir(sequence_path):
                    frame_count = len([f for f in os.listdir(sequence_path) if f.endswith('.png')])
                    if frame_count >= 8:
                        print(f"Subject: {subject_str}, Sequence: {sequence}, Frames: {frame_count}")
                        sequence_path = os.path.join(subject_path, sequence)
                        image_paths = get_image_paths(sequence_path)
                        n = len(image_paths)
                        # Step 3: Compute correlations with the first average image
                        correlations = compute_correlations(image_paths, avg_image)
                        smoothed_correlations = smooth_signal(correlations, window_size=3)
                        mean_correlation = np.mean(correlations)
                        print("correlations length: ", len(correlations))
                        print("smoothed_correlations length: ", len(smoothed_correlations))
                        print("mean_correlation: ", mean_correlation)


                        # Step 5: Find peaks and select three middle peaks
                        smoothed_peaks, _ = find_peaks(smoothed_correlations)
                        nearest_peaks = find_nearest_peaks(correlations, smoothed_peaks)

                        print(f"Smoothed peaks: {smoothed_peaks}")
                        print(f"Nearest peaks in the original signal: {nearest_peaks}")

                        m = len(nearest_peaks)
                        if m >= 5:
                            mid_idx = m // 2
                            middle_peaks = [nearest_peaks[mid_idx - 2], nearest_peaks[mid_idx], nearest_peaks[mid_idx + 2]]
                            start_frame = middle_peaks[0]
                            end_frame = middle_peaks[2]
                            selected_frames = image_paths[start_frame:end_frame + 1]

                            print(
                                f"Selected frame indices for the second average: {list(range(start_frame, end_frame + 1))}")
                        else:
                            if m >= 3:
                                mid_idx = m // 2
                                middle_peaks = [nearest_peaks[mid_idx - 1], nearest_peaks[mid_idx], nearest_peaks[mid_idx + 1]]
                                start_frame = middle_peaks[0]
                                end_frame = middle_peaks[2]
                                selected_frames = image_paths[start_frame:end_frame + 1]

                                print(
                                    f"Selected frame indices for the second average: {list(range(start_frame, end_frame + 1))}")
                            else:
                                selected_frames = image_paths
                                print("Not enough peaks detected, using all frames.")

                        # Step 6: Compute a new average frame
                        new_avg_image = compute_average_image(selected_frames)
                        print("Computed new average image from selected frames.")
                        # Step 7: Compute correlations with the new average image
                        correlations_new_avg = compute_correlations(image_paths, new_avg_image)
                        # Step 8: Plot the correlation coefficients, smoothed values, and mean correlation
                        smoothed_new_avg = smooth_signal(correlations_new_avg, window_size=3)
                        mean_correlation_new = np.mean(correlations_new_avg)
                        std_correlation_new = np.std(correlations_new_avg)
                        mean_smoothed_new = np.mean(smoothed_new_avg)
                        std_smoothed_new = np.std(smoothed_new_avg)

                        # Find peaks in correlations and smoothed correlations
                        peaks_corr, _ = find_peaks(correlations_new_avg)
                        peaks_smooth, _ = find_peaks(smoothed_new_avg)

                        print(f"Mean correlation: {mean_correlation_new}")
                        print(f"Standard deviation of correlation: {std_correlation_new}")

                        print(f"Smoothed correlations length: {len(smoothed_new_avg)}")
                        print(f"Correlations with new average image length: {len(correlations_new_avg)}")

                        # Filter peaks where value > (mean - std deviation)
                        valid_peaks_corr = [p for p in peaks_corr if
                                            correlations_new_avg[p] >= (mean_correlation_new - std_correlation_new*0.5)]
                        valid_peaks_smooth = [p for p in peaks_smooth if
                                              smoothed_new_avg[p] >= (mean_smoothed_new - std_smoothed_new*0.5)]

                        print(f"valid_peaks_corr : {valid_peaks_corr}")
                        print(f"valid_peaks_smooth : {valid_peaks_smooth}")

                        # Get the nearest peaks in the original correlation signal based on valid smoothed peaks
                        nearest_valid_peaks_corr = find_nearest_peaks1(valid_peaks_corr, valid_peaks_smooth)
                        print(f"Nearest valid peaks in non-smoothed correlation signal: {nearest_valid_peaks_corr}")
                        if nearest_valid_peaks_corr:
                            start_frame = nearest_valid_peaks_corr[0]  # First peak
                            if start_frame>frame_count*0.22:
                                start_frame = int(frame_count*0.22)

                            end_frame = nearest_valid_peaks_corr[-1]  # Last peak
                            if end_frame<(frame_count*0.78):
                                end_frame = int(frame_count*0.78)

                            # Select frames between first and last peak
                            selected_frames = image_paths[start_frame:end_frame + 1]

                            # Define output directory
                            output_dir = os.path.join(save_folder, subject_str, sequence)
                            os.makedirs(output_dir, exist_ok=True)

                            # Save frames
                            for frame_path in selected_frames:
                                shutil.copy(frame_path, output_dir)

                            print(
                                f"Saved {len(selected_frames)} frames from {start_frame} to {end_frame} in {output_dir}")
                        else:
                            print("No valid peaks found, no frames saved.")


def global_average_image(root_folder):
    all_images = []

    subject_folders = sorted(
        [os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])

    for subject_index, subject_path in enumerate(subject_folders):
        if subject_index >= 30:
            break  # Stop at first 30 subjects
        print(f"\nProcessing Subject {subject_index + 1}: {os.path.basename(subject_path)}")

        sequence_folders = sorted([os.path.join(subject_path, seq) for seq in os.listdir(subject_path) if
                                   os.path.isdir(os.path.join(subject_path, seq))])

        for sequence_index, seq_path in enumerate(sequence_folders):
            print(f"  Sequence {sequence_index + 1}: {os.path.basename(seq_path)}")
            image_paths = sorted([os.path.join(seq_path, img) for img in os.listdir(seq_path) if
                                  img.lower().endswith(('.png', '.jpg', '.jpeg'))])
            n = len(image_paths)

            if n > 59:
                start_idx = n // 3
                end_idx = 2 * n // 3

                for i in range(start_idx, end_idx):
                    img = read_image(image_paths[i])
                    if img is not None:
                        all_images.append(img)

    if all_images:
        avg_image = np.mean(all_images, axis=0).astype(np.uint8)
        return avg_image
    else:
        print("No images found matching criteria.")
        return None

# Set your root folder path
root_folder = r"E:\Gait_IIT_BHU_Data\Merged_Normalized_Aligned_Images"
save_folder = r"E:\Gait_IIT_BHU_Data\Normalized_Aligned_Refined_Images"
# Run the function
average_img = global_average_image(root_folder)

# Save or display the result
if average_img is not None:
    cv2.imwrite(r"E:\Gait_IIT_BHU_Data\global_average.png", average_img)
    print("Global average image saved.")
else:
    print("Global average image not computed.")

count_frames_per_sequence(root_folder, save_folder, average_img)