import cv2
import argparse
from enhanced_lip_detector import EnhancedLipDetector

def process_video(input_path, output_path):
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize detector
    detector = EnhancedLipDetector()
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, speaking_status = detector.detect_speaking(frame)
        
        # Write frame
        out.write(processed_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Test Enhanced Lip Detector")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output video")
    
    args = parser.parse_args()
    
    process_video(args.input, args.output)

if __name__ == "__main__":
    main() 