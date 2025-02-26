import cv2
from lip_detector import LipDetector
import argparse
import os

def process_video(input_path, output_path):
    # Convert paths to absolute paths with correct separators
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    
    # Ensure input directory exists
    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}")
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize detector
    detector = LipDetector()
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))  # Convert to string explicitly
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_path}")
        return
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print(f"Processing video: {input_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        frame, speaking_status = detector.detect_speaking(frame)
        
        # Display overall status
        y_pos = 30
        for face_id, is_speaking in speaking_status.items():
            status = f"Face {face_id+1}: {'Speaking' if is_speaking else 'Silent'}"
            cv2.putText(frame, status, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0) if is_speaking else (0, 0, 255), 1)
            y_pos += 20
        
        # Write frame
        out.write(frame)
        
        # Update progress
        frame_count += 1
        if frame_count % 30 == 0:  # Update every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%", end='\r')
        
        # Optional: Display frame while processing
        cv2.imshow('Processing', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete! Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process video file for lip movement detection')
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('-o', '--output', help='Path to output video file')
    
    args = parser.parse_args()
    
    # Convert backslashes to forward slashes and handle relative paths
    input_path = os.path.normpath(args.input)
    
    # If output path not specified, create one in same directory as input
    if args.output is None:
        input_dir = os.path.dirname(input_path)
        filename = os.path.splitext(os.path.basename(input_path))[0]
        args.output = os.path.join(input_dir, f"{filename}_processed.mp4")
    
    process_video(input_path, args.output)

if __name__ == "__main__":
    main() 