import cv2
import numpy as np
import os

input_root_dir = f'E:\\Gait_IIT_BHU_Data\\Sequences_with_Labeled_Direction\\'
output_root_dir = f'E:\\Gait_IIT_BHU_Data\\Images_With_Moving_Objects\\'
# Ensure the output root directory exists
os.makedirs(output_root_dir, exist_ok=True)
# Iterate over the hardcoded range of subjects (201 to 333)
for subject in range(1, 334):
    subject_id = f"{subject:03d}"
    subject_str = str(subject_id)  # Convert subject number to string
    subject_path = os.path.join(input_root_dir, subject_str)
    # Ensure it is a directory
    if not os.path.isdir(subject_path):
        continue

    print(f"Processing subject: {subject_str}")

    # Iterate over each sequence (folders inside each subject)
    for sequence in sorted(os.listdir(subject_path)):
        sequence_path = os.path.join(subject_path, sequence)

        # Ensure it is a directory
        if not os.path.isdir(sequence_path):
            continue

        print(f"  Processing sequence: {sequence}")

        # Create the corresponding output directory
        output_dir = os.path.join(output_root_dir, subject_str, sequence)
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Load all images in the sequence
        image_files = sorted([f for f in os.listdir(sequence_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Read images into a list
        image_list = []
        for file in image_files:
            img_path = os.path.join(sequence_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                image_list.append(img)

        # Skip processing if there are no valid images
        if not image_list:
            print(f"  ⚠️ No valid images found in {sequence_path}. Skipping...")
            continue

        # Convert list to NumPy array for easy processing
        image_stack = np.array(image_list, dtype=np.uint8)

        # Step 2: Compute the background model (median of all frames)
        background = np.median(image_stack, axis=0).astype(np.uint8)

        # Step 3: Remove static objects by subtracting background from each frame
        for i, img in enumerate(image_stack):
            moving_objects = cv2.absdiff(img, background)  # Compute absolute difference
            _, moving_objects = cv2.threshold(moving_objects, 30, 255, cv2.THRESH_BINARY)  # Threshold to binary

            # Save the result
            output_path = os.path.join(output_dir, image_files[i])
            cv2.imwrite(output_path, moving_objects)

        print(f"  ✅ Completed processing for {sequence}")

print("🎉 Processing completed for all subjects and sequences!")