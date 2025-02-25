import os
import shutil
import re

def organize_grid_videos():
    # Source and destination paths
    downloads_dir = os.path.expanduser("~/Downloads")
    dataset_dir = "dataset"

    # Regular expression to match speaker folders (s1, s2, etc.)
    speaker_pattern = re.compile(r's(\d+)\.mpg_vcd')

    # Find all speaker folders in Downloads
    for item in os.listdir(downloads_dir):
        match = speaker_pattern.match(item)
        if match:
            speaker_num = int(match.group(1))
            
            # Skip speaker 21 as it has no data
            if speaker_num == 21:
                print(f"Skipping speaker {speaker_num} (no data available)")
                continue
                
            speaker_folder = os.path.join(downloads_dir, item, f"s{speaker_num}")
            
            # Determine destination based on speaker number
            if speaker_num <= 20:
                dest_base = os.path.join(dataset_dir, "train")
            elif speaker_num <= 28:
                dest_base = os.path.join(dataset_dir, "val")
            else:
                dest_base = os.path.join(dataset_dir, "test")
            
            # Create destination folder
            dest_folder = os.path.join(dest_base, "speaking", "GRID", str(speaker_num))
            os.makedirs(dest_folder, exist_ok=True)
            
            # Copy and rename .mpg files sequentially
            if os.path.exists(speaker_folder):
                video_count = 1
                for video in os.listdir(speaker_folder):
                    if video.endswith('.mpg'):
                        src = os.path.join(speaker_folder, video)
                        new_name = f"{video_count:03d}.mpg"  # Format: video_001.mpg
                        dst = os.path.join(dest_folder, new_name)
                        print(f"Copying {src} to {dst}")
                        shutil.copy2(src, dst)
                        video_count += 1
                print(f"Processed {video_count-1} videos for speaker {speaker_num}")

if __name__ == "__main__":
    print("Starting to organize GRID videos...")
    organize_grid_videos()
    print("Done organizing videos!")