import cv2
import os
import numpy as np
from tqdm import tqdm
import dlib

def get_lip_height(shape):
    # Get points for inner lip
    top_lip = np.array([shape.part(61).y, shape.part(62).y, shape.part(63).y])
    bottom_lip = np.array([shape.part(67).y, shape.part(66).y, shape.part(65).y])
    
    # Calculate average separation between top and bottom lip
    return np.mean(np.abs(top_lip - bottom_lip))

def analyze_grid_videos(threshold_factor=0.080):  # Reduced from 0.5 to 0.25 to be more sensitive
    dataset_dir = "dataset"
    sample_videos = []
    
    # Initialize dlib's face detector and facial landmarks predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    
    # Get a sample from each split
    for split in ['train', 'val', 'test']:
        path = os.path.join(dataset_dir, split, 'speaking', 'GRID')
        if os.path.exists(path):
            for speaker in os.listdir(path):
                speaker_path = os.path.join(path, speaker)
                if os.path.isdir(speaker_path):
                    videos = [f for f in os.listdir(speaker_path) if f.endswith('.mpg')]
                    if videos:
                        sample_videos.append((split, os.path.join(speaker_path, videos[0])))
                        break

    print(f"Analyzing {len(sample_videos)} sample videos...")
    print(f"Using threshold factor: {threshold_factor}")
    
    for split, video_path in tqdm(sample_videos):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"\nVideo: {video_path}")
        print(f"Split: {split}")
        print(f"Total frames: {frame_count}")
        print(f"FPS: {fps}")
        
        # Save frames by classification
        output_dir = os.path.join("analysis_output", f"{split}_{os.path.basename(video_path).replace('.mpg', '')}")
        speaking_dir = os.path.join(output_dir, "speaking")
        silent_dir = os.path.join(output_dir, "silent")
        os.makedirs(speaking_dir, exist_ok=True)
        os.makedirs(silent_dir, exist_ok=True)
        
        # Analyze each frame
        lip_heights = []
        frames = []
        
        print("Processing frames...")
        for i in range(frame_count):
            ret, frame = cap.read()
            if ret:
                # Detect face
                faces = detector(frame, 0)
                if len(faces) > 0:
                    # Get facial landmarks
                    shape = predictor(frame, faces[0])
                    lip_height = get_lip_height(shape)
                    lip_heights.append(lip_height)
                    frames.append(frame)
                else:
                    lip_heights.append(0)
                    frames.append(frame)
        
        # Calculate threshold for speaking vs silent
        lip_heights = np.array(lip_heights)
        min_height = np.min(lip_heights)
        max_height = np.max(lip_heights)
        threshold = min_height + (max_height - min_height) * threshold_factor
        
        print(f"Lip height stats:")
        print(f"Min: {min_height:.2f}")
        print(f"Max: {max_height:.2f}")
        print(f"Mean: {np.mean(lip_heights):.2f}")
        print(f"Threshold: {threshold:.2f}")
        
        # Save frames with classification
        speaking_count = 0
        silent_count = 0
        
        for i, (frame, height) in enumerate(zip(frames, lip_heights)):
            if height > threshold:
                cv2.imwrite(os.path.join(speaking_dir, f'frame_{i:03d}.jpg'), frame)
                speaking_count += 1
            else:
                cv2.imwrite(os.path.join(silent_dir, f'frame_{i:03d}.jpg'), frame)
                silent_count += 1
        
        cap.release()
        
        print(f"Saved classified frames to {output_dir}")
        print(f"Speaking frames: {speaking_count}")
        print(f"Silent frames: {silent_count}")

if __name__ == "__main__":
    # You can adjust this value to be more sensitive to lip movement
    # Lower values = more frames classified as speaking
    # Higher values = more frames classified as silent
    threshold_factor = 0.10  # Try different values: 0.25, 0.15, 0.1
    
    analyze_grid_videos(threshold_factor)
    print("\nAnalysis complete! Check the analysis_output directory.")
    print("Frames are sorted into 'speaking' and 'silent' subdirectories.")
    print("Look through them to verify the classification.")