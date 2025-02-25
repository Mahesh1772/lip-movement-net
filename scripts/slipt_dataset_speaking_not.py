import os
import numpy as np
from tqdm import tqdm
import dlib
from moviepy.editor import VideoFileClip
import cv2

def get_lip_height(shape):
    top_lip = np.array([shape.part(61).y, shape.part(62).y, shape.part(63).y])
    bottom_lip = np.array([shape.part(67).y, shape.part(66).y, shape.part(65).y])
    return np.mean(np.abs(top_lip - bottom_lip))

def split_videos(threshold_factor=0.10):
    dataset_dir = "dataset"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    
    # Process each split
    for split in ['train', 'val', 'test']:
        speaking_path = os.path.join(dataset_dir, split, 'speaking', 'GRID')
        silent_path = os.path.join(dataset_dir, split, 'silent', 'GRID')
        
        if os.path.exists(speaking_path):
            print(f"\nProcessing {split} split...")
            
            # Process each speaker
            for speaker in os.listdir(speaking_path):
                speaker_speaking_path = os.path.join(speaking_path, speaker)
                speaker_silent_path = os.path.join(silent_path, speaker)
                os.makedirs(speaker_silent_path, exist_ok=True)
                
                if os.path.isdir(speaker_speaking_path):
                    videos = [f for f in os.listdir(speaker_speaking_path) if f.endswith('.mpg')]
                    print(f"Processing {len(videos)} videos for speaker {speaker}")
                    
                    for video_name in tqdm(videos):
                        try:
                            video_path = os.path.join(speaker_speaking_path, video_name)
                            
                            # Load video using MoviePy
                            clip = VideoFileClip(video_path)
                            
                            # Process frames
                            lip_heights = []
                            speaking_frames = []
                            silent_frames = []
                            
                            for frame in clip.iter_frames():
                                # Convert from RGB to BGR for dlib
                                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                faces = detector(frame_bgr, 0)
                                
                                if len(faces) > 0:
                                    shape = predictor(frame_bgr, faces[0])
                                    lip_height = get_lip_height(shape)
                                    lip_heights.append(lip_height)
                                    
                                    # Store original RGB frame
                                    if lip_height > 0:  # Valid detection
                                        speaking_frames.append(frame) if lip_height > threshold else silent_frames.append(frame)
                            
                            clip.close()
                            
                            if not lip_heights:
                                continue
                                
                            # Calculate threshold
                            lip_heights = np.array(lip_heights)
                            min_height = np.min(lip_heights)
                            max_height = np.max(lip_heights)
                            threshold = min_height + (max_height - min_height) * threshold_factor
                            
                            # Save speaking frames
                            if speaking_frames:
                                speaking_clip = VideoFileClip(video_path).set_frames(speaking_frames)
                                speaking_clip.write_videofile(
                                    os.path.join(speaker_speaking_path, video_name),
                                    codec='mpeg1video',
                                    audio=False
                                )
                                speaking_clip.close()
                            
                            # Save silent frames
                            if silent_frames:
                                silent_clip = VideoFileClip(video_path).set_frames(silent_frames)
                                silent_clip.write_videofile(
                                    os.path.join(speaker_silent_path, video_name),
                                    codec='mpeg1video',
                                    audio=False
                                )
                                silent_clip.close()
                            
                        except Exception as e:
                            print(f"Error processing {video_path}: {str(e)}")
                            continue

if __name__ == "__main__":
    # Install moviepy if not already installed
    try:
        import moviepy
    except ImportError:
        print("Installing required package: moviepy")
        import subprocess
        subprocess.check_call(["pip", "install", "moviepy"])
        
    split_videos()
    print("\nAnalysis complete!")