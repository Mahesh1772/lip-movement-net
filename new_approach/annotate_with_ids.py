import cv2
import numpy as np
import argparse
import os
import json
import time
from lip_detector import LipDetector

class FaceAnnotator:
    def __init__(self, video_path, output_json, output_dir, sample_rate=30):
        """
        Initialize the face annotator.
        
        Args:
            video_path: Path to input video
            output_json: Path to save ground truth JSON
            output_dir: Directory to save visualization frames
            sample_rate: Sample every N frames
        """
        self.video_path = video_path
        self.output_json = output_json
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        
        # Initialize detector
        self.detector = LipDetector()
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output directories
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load existing ground truth if available
        self.ground_truth = {}
        if os.path.exists(output_json):
            with open(output_json, 'r') as f:
                self.ground_truth = json.load(f)
            print(f"Loaded existing ground truth from {output_json}")
    
    def process_video(self):
        """Process the video and create visualization frames with face IDs."""
        print(f"Processing video: {self.video_path}")
        print(f"Total frames: {self.total_frames}")
        print(f"Sampling every {self.sample_rate} frames")
        
        # Process frames
        frame_indices = list(range(0, self.total_frames, self.sample_rate))
        
        for i, frame_idx in enumerate(frame_indices):
            # Load frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                print(f"Error reading frame {frame_idx}")
                continue
            
            # Process frame to get faces
            _, speaking_status = self.detector.detect_speaking(frame.copy())
            
            # Get face data for visualization
            faces_data = self.detector.get_faces_and_lips(frame)
            
            # Create visualization frame
            vis_frame = frame.copy()
            
            # Add frame information
            cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw face IDs and bounding boxes
            for face_data in faces_data:
                face_id = face_data['id']
                bbox = face_data['bbox']
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Use different colors for different face IDs
                color_id = (face_id * 50) % 255
                color = (color_id, 255, 255 - color_id)
                
                # Draw rectangle
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Get speaking status from detector or ground truth
                frame_key = str(frame_idx)
                face_key = str(face_id)
                
                is_speaking = False
                if frame_key in self.ground_truth and face_key in self.ground_truth[frame_key]:
                    is_speaking = self.ground_truth[frame_key][face_key]
                elif face_id in speaking_status:
                    is_speaking = speaking_status[face_id]
                
                # Save to ground truth
                if frame_key not in self.ground_truth:
                    self.ground_truth[frame_key] = {}
                self.ground_truth[frame_key][face_key] = is_speaking
                
                # Draw face ID with large, visible text
                face_id_text = f"Face {face_id}"
                cv2.putText(vis_frame, face_id_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Draw speaking status
                status = "Speaking" if is_speaking else "Silent"
                cv2.putText(vis_frame, status, (x1, y2 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0) if is_speaking else (0, 0, 255), 2)
            
            # Save visualization frame
            output_path = os.path.join(self.output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(output_path, vis_frame)
            
            # Save ground truth periodically
            if i % 10 == 0:
                self.save_ground_truth()
            
            # Show progress
            progress = (i + 1) / len(frame_indices) * 100
            print(f"Progress: {progress:.1f}% complete", end='\r')
        
        # Final save
        self.save_ground_truth()
        
        # Cleanup
        self.cap.release()
        
        print(f"\nProcessing complete!")
        print(f"Ground truth saved to: {self.output_json}")
        print(f"Visualization frames saved to: {self.output_dir}")
    
    def save_ground_truth(self):
        """Save the ground truth data to file."""
        with open(self.output_json, 'w') as f:
            json.dump(self.ground_truth, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Create visualization frames with face IDs')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output-json', required=True, help='Path to save ground truth JSON')
    parser.add_argument('--output-dir', required=True, help='Directory to save visualization frames')
    parser.add_argument('--sample-rate', type=int, default=30, 
                        help='Sample every N frames (default: 30)')
    
    args = parser.parse_args()
    
    annotator = FaceAnnotator(args.video, args.output_json, args.output_dir, args.sample_rate)
    annotator.process_video()

if __name__ == "__main__":
    main() 