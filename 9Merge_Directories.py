import os
import shutil
import re

def extract_frame_number(filename):
    """Extracts frame number from the filename."""
    match = re.search(r'frame(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def extract_direction(sequence_name):
    """Extracts the direction (right_to_left or left_to_right) from the sequence name."""
    if "right_to_left" in sequence_name.lower():
        return "right_to_left"
    elif "left_to_right" in sequence_name.lower():
        return "left_to_right"
    else:
        return None

def merge_sequences_and_save(input_subject_dir, output_subject_dir):
    """Merge sequences and save into a different output directory."""
    sequences = sorted(os.listdir(input_subject_dir))
    os.makedirs(output_subject_dir, exist_ok=True)

    i = 0

    while i < len(sequences):
        current_sequence = sequences[i]
        current_sequence_path = os.path.join(input_subject_dir, current_sequence)

        if not os.path.isdir(current_sequence_path):
            i += 1
            continue

        current_frames = sorted(os.listdir(current_sequence_path))
        if not current_frames or len(current_frames) < 4:
            print(f"⚠️ Skipping sequence with < 4 frames: {current_sequence}")
            i += 1
            continue

        current_direction = extract_direction(current_sequence)

        # Name merged sequence same as the current sequence
        merged_sequence_name = current_sequence
        merged_sequence_path = os.path.join(output_subject_dir, merged_sequence_name)
        os.makedirs(merged_sequence_path, exist_ok=True)

        # Copy all frames from the current sequence
        for frame in current_frames:
            src = os.path.join(current_sequence_path, frame)
            dst = os.path.join(merged_sequence_path, frame)
            shutil.copy(src, dst)

        # Try merging with next sequences if frame gap is small and direction matches
        while i < len(sequences) - 1:
            next_sequence = sequences[i + 1]
            next_sequence_path = os.path.join(input_subject_dir, next_sequence)

            if not os.path.isdir(next_sequence_path):
                i += 1
                continue

            next_frames = sorted(os.listdir(next_sequence_path))
            if not next_frames or len(next_frames) < 4:
                print(f"⚠️ Skipping next sequence with < 4 frames: {next_sequence}")
                i += 1
                continue

            next_direction = extract_direction(next_sequence)

            last_frame_current = extract_frame_number(current_frames[-1])
            first_frame_next = extract_frame_number(next_frames[0])

            if last_frame_current is None or first_frame_next is None:
                break

            # ➡️ Merge only if direction matches and frame gap < 4
            if (current_direction == next_direction) and (0 < (first_frame_next - last_frame_current) < 4):
                print(f"🛠️ Merging {next_sequence} into {merged_sequence_name} (gap={first_frame_next - last_frame_current}, direction={current_direction})")

                for frame in next_frames:
                    src = os.path.join(next_sequence_path, frame)
                    dst = os.path.join(merged_sequence_path, frame)
                    shutil.copy(src, dst)

                current_frames.extend(next_frames)
                i += 1  # Proceed to the next one after merged
            else:
                break

        i += 1



input_root_dir = f"E:\\Gait_IIT_BHU_Data\\Normalized_Aligned_Images" #E:\Gait_IIT_BHU_Data\Normalized_Aligned_Images
output_root_dir = f"E:\\Gait_IIT_BHU_Data\\Merged_Normalized_Aligned_Images"


for subject_id in range(1, 134):  # Subject IDs 201 to 235
    subject = f"{subject_id:03d}"
    subject_str = str(subject)
    input_subject_dir = os.path.join(input_root_dir, subject_str)
    output_subject_dir = os.path.join(output_root_dir, subject_str)

    if not os.path.isdir(input_subject_dir):
        print(f"⚠️ Skipping missing subject folder: {input_subject_dir}")
        continue

    print(f"\n📂 Processing Subject {subject_str}...")
    merge_sequences_and_save(input_subject_dir, output_subject_dir)
print("\n✅ Done merging all subjects!")