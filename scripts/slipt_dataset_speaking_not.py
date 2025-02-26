import os
import numpy as np
from tqdm import tqdm
import dlib
import cv2
import subprocess
import tempfile
import glob

def get_lip_height(shape):
    top_lip = np.array([shape.part(61).y, shape.part(62).y, shape.part(63).y])
    bottom_lip = np.array([shape.part(67).y, shape.part(66).y, shape.part(65).y])
    return np.mean(np.abs(top_lip - bottom_lip))

def check_video_validity(video_path):
    """Check if the video file is valid and can be read by OpenCV"""
    cap = cv2.VideoCapture(video_path)
    valid = cap.isOpened()
    cap.release()
    return valid

def process_mpg_file(video_path):
    """
    Process MPG file by first extracting frames using FFmpeg directly
    """
    try:
        # Create temp directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract frames using FFmpeg (ignore audio)
            subprocess.run([
                'ffmpeg', '-y',
                '-i', video_path,
                '-an',  # Remove audio
                '-vf', 'fps=25',  # Force 25 fps
                os.path.join(temp_dir, 'frame_%05d.jpg')
            ], check=True, capture_output=True)
            
            # Read frames using OpenCV
            frames = []
            frame_files = sorted(glob.glob(os.path.join(temp_dir, 'frame_*.jpg')))
            
            if not frame_files:
                return None, None, None, None
                
            # Read first frame to get dimensions
            first_frame = cv2.imread(frame_files[0])
            if first_frame is None:
                return None, None, None, None
                
            height, width = first_frame.shape[:2]
            frames.append(first_frame)
            
            # Read rest of the frames
            for frame_file in frame_files[1:]:
                frame = cv2.imread(frame_file)
                if frame is not None:
                    frames.append(frame)
            
            return frames, width, height, 25  # Fixed 25 fps for GRID corpus
            
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error processing {video_path}: {e.stderr.decode()}")
        return None, None, None, None
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return None, None, None, None

def split_videos(threshold_factor=0.10):
    dataset_dir = "dataset"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    
    for split in ['train', 'val', 'test']:
        speaking_path = os.path.join(dataset_dir, split, 'speaking', 'GRID')
        silent_path = os.path.join(dataset_dir, split, 'silent', 'GRID')
        
        # Create output directories with _mp4 suffix
        speaking_path_mp4 = speaking_path + '_mp4'
        silent_path_mp4 = silent_path + '_mp4'
        os.makedirs(speaking_path_mp4, exist_ok=True)
        os.makedirs(silent_path_mp4, exist_ok=True)
        
        if os.path.exists(speaking_path):
            print(f"\nProcessing {split} split...")
            
            for speaker in os.listdir(speaking_path):
                speaker_speaking_path = os.path.join(speaking_path, speaker)
                speaker_silent_path = os.path.join(silent_path, speaker)
                
                # Create MP4 output directories
                speaker_speaking_path_mp4 = os.path.join(speaking_path_mp4, speaker)
                speaker_silent_path_mp4 = os.path.join(silent_path_mp4, speaker)
                os.makedirs(speaker_speaking_path_mp4, exist_ok=True)
                os.makedirs(speaker_silent_path_mp4, exist_ok=True)
                
                if os.path.isdir(speaker_speaking_path):
                    videos = [f for f in os.listdir(speaker_speaking_path) if f.endswith('.mpg')]
                    print(f"Processing {len(videos)} videos for speaker {speaker}")
                    
                    for video_name in tqdm(videos):
                        try:
                            video_path = os.path.join(speaker_speaking_path, video_name)
                            
                            # Verify the file exists before processing
                            if not os.path.exists(video_path):
                                print(f"File not found: {video_path}")
                                continue
                            
                            # Process the MPG file
                            frames, width, height, fps = process_mpg_file(video_path)
                            
                            if frames is None:
                                print(f"Cannot process {video_path} - could not extract frames")
                                continue
                            
                            # Process frames for lip detection
                            lip_heights = []
                            valid_frames = []
                            
                            for frame in frames:
                                # Check if frame is valid
                                if frame is None or frame.size == 0:
                                    continue
                                    
                                # Convert to grayscale for better face detection
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                
                                # Detect faces
                                faces = detector(gray, 0)
                                if len(faces) > 0:
                                    shape = predictor(gray, faces[0])
                                    lip_height = get_lip_height(shape)
                                    lip_heights.append(lip_height)
                                    valid_frames.append(frame)
                            
                            if not lip_heights or not valid_frames:
                                print(f"No valid faces detected in {video_path}")
                                continue
                            
                            # Calculate threshold
                            lip_heights = np.array(lip_heights)
                            min_height = np.min(lip_heights)
                            max_height = np.max(lip_heights)
                            threshold = min_height + (max_height - min_height) * threshold_factor
                            
                            # Split frames
                            speaking_frames = []
                            silent_frames = []
                            
                            for frame, height in zip(valid_frames, lip_heights):
                                if height > threshold:
                                    speaking_frames.append(frame)
                                else:
                                    silent_frames.append(frame)
                            
                            # Ensure we have the correct dimensions for output
                            frame_height, frame_width = valid_frames[0].shape[:2]
                            
                            # Save speaking frames as MP4
                            if speaking_frames:
                                output_path = os.path.join(speaker_speaking_path_mp4, 
                                                          os.path.splitext(video_name)[0] + '.mp4')
                                out = cv2.VideoWriter(
                                    output_path,
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    fps,
                                    (frame_width, frame_height)
                                )
                                for frame in speaking_frames:
                                    out.write(frame)
                                out.release()
                            
                            # Save silent frames as MP4
                            if silent_frames:
                                output_path = os.path.join(speaker_silent_path_mp4, 
                                                          os.path.splitext(video_name)[0] + '.mp4')
                                out = cv2.VideoWriter(
                                    output_path,
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    fps,
                                    (frame_width, frame_height)
                                )
                                for frame in silent_frames:
                                    out.write(frame)
                                out.release()
                            
                        except Exception as e:
                            print(f"Error processing {video_path}: {str(e)}")
                            continue

def main():
    # Make sure the model directory exists
    os.makedirs("models", exist_ok=True)
    
    # Check if the shape predictor file exists
    predictor_path = "models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        print(f"Warning: Shape predictor file not found at {predictor_path}")
        print("Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract it and place it in the models directory")
        return
    
    split_videos()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()