import cv2
import argparse
from lip_detector import MediaPipeLipDetector
from tqdm import tqdm

def process_video(input_path, output_path):
    # Initialize detector
    detector = MediaPipeLipDetector()
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame with tqdm progress bar
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, speaking_status = detector.detect_speaking(frame)
            
            # Write to output video
            out.write(processed_frame)
            
            # Update progress bar
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process video for lip movement detection')
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--output', required=True, help='Output video path')
    args = parser.parse_args()
    
    process_video(args.input, args.output)

if __name__ == "__main__":
    main() 