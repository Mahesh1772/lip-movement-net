import cv2
import numpy as np
import mediapipe as mp
import insightface
from insightface.app import FaceAnalysis
import argparse
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class LipDetectionDiagnostic:
    def __init__(self):
        # Initialize InsightFace for face detection
        self.app = FaceAnalysis(
            allowed_modules=['detection'], 
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize MediaPipe for lip landmarks
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,  # Set to True for image analysis
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.3,  # Lower threshold for better detection in wide shots
            min_tracking_confidence=0.3
        )
        
        # MediaPipe lip indices
        self.UPPER_LIP_INDICES = [13, 14, 312]  # Upper lip indices in MediaPipe
        self.LOWER_LIP_INDICES = [17, 16, 15]   # Lower lip indices in MediaPipe
        
        # InsightFace lip indices (if available)
        self.IF_UPPER_LIP_INDICES = [62, 63, 64]  # Upper lip indices in InsightFace
        self.IF_LOWER_LIP_INDICES = [66, 67, 68]  # Lower lip indices in InsightFace
    
    def is_wide_angle_shot(self, image, faces):
        """Determine if the current frame is a wide angle shot"""
        if not faces:
            return False
        
        # Calculate face sizes relative to frame
        frame_height, frame_width = image.shape[:2]
        frame_area = frame_height * frame_width
        
        max_face_ratio = 0
        
        for face in faces:
            bbox = face.bbox
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            face_area = face_width * face_height
            face_ratio = face_area / frame_area
            
            # Keep track of the largest face
            max_face_ratio = max(max_face_ratio, face_ratio)
        
        # Use face size as the criterion
        is_wide = max_face_ratio < 0.02
        
        return is_wide, max_face_ratio
    
    def analyze_image(self, image_path, output_dir=None):
        """Analyze a single image and visualize lip detection results"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Get faces using InsightFace
        faces = self.app.get(image)
        print(f"Found {len(faces)} faces using InsightFace")
        
        # Check if it's a wide angle shot
        is_wide, max_face_ratio = self.is_wide_angle_shot(image, faces)
        print(f"Shot type: {'Wide angle' if is_wide else 'Close-up'}")
        print(f"Max face ratio: {max_face_ratio:.6f}")
        
        # Process with MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_results = self.mp_face_mesh.process(rgb_image)
        
        # Create visualization image
        vis_image = image.copy()
        
        # Draw face boxes from InsightFace
        for i, face in enumerate(faces):
            bbox = face.bbox
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, f"Face {i}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Print face dimensions
            face_width = x2 - x1
            face_height = y2 - y1
            print(f"Face {i}: Size = {face_width}x{face_height}, Area ratio = {(face_width*face_height)/(image.shape[0]*image.shape[1]):.6f}")
        
        # Check MediaPipe results
        if mp_results.multi_face_landmarks:
            print(f"Found {len(mp_results.multi_face_landmarks)} faces using MediaPipe")
            
            # Draw MediaPipe landmarks
            for i, face_landmarks in enumerate(mp_results.multi_face_landmarks):
                # Draw all lip landmarks
                lip_indices = self.UPPER_LIP_INDICES + self.LOWER_LIP_INDICES
                
                # Draw larger dots for better visibility
                for idx in lip_indices:
                    x = int(face_landmarks.landmark[idx].x * image.shape[1])
                    y = int(face_landmarks.landmark[idx].y * image.shape[0])
                    cv2.circle(vis_image, (x, y), 4, (0, 0, 255), -1)
                    
                # Calculate lip height
                h, w = image.shape[:2]
                upper_lip_y = np.mean([face_landmarks.landmark[idx].y for idx in self.UPPER_LIP_INDICES]) * h
                lower_lip_y = np.mean([face_landmarks.landmark[idx].y for idx in self.LOWER_LIP_INDICES]) * h
                lip_height = abs(upper_lip_y - lower_lip_y)
                
                # Normalize by face height
                nose_y = face_landmarks.landmark[1].y * h  # Nose tip
                chin_y = face_landmarks.landmark[152].y * h  # Chin
                face_height = abs(nose_y - chin_y)
                
                normalized_height = lip_height / face_height if face_height > 0 else 0.0
                print(f"MediaPipe Face {i}: Lip height = {lip_height:.2f}px, Normalized = {normalized_height:.6f}")
        else:
            print("No MediaPipe face landmarks detected!")
        
        # Check InsightFace 3D landmarks
        for i, face in enumerate(faces):
            if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
                print(f"InsightFace 3D landmarks available for Face {i}")
                
                # Draw InsightFace lip landmarks if available
                landmarks = face.landmark_3d_68
                for idx in self.IF_UPPER_LIP_INDICES + self.IF_LOWER_LIP_INDICES:
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(vis_image, (x, y), 4, (255, 0, 0), -1)
                
                # Calculate lip height using InsightFace
                upper_lip_y = np.mean([landmarks[idx][1] for idx in self.IF_UPPER_LIP_INDICES])
                lower_lip_y = np.mean([landmarks[idx][1] for idx in self.IF_LOWER_LIP_INDICES])
                lip_height = abs(upper_lip_y - lower_lip_y)
                
                # Normalize by face height
                nose_y = landmarks[30][1]  # Nose tip
                chin_y = landmarks[8][1]   # Chin
                face_height = abs(nose_y - chin_y)
                
                normalized_height = lip_height / face_height if face_height > 0 else 0.0
                print(f"InsightFace Face {i}: Lip height = {lip_height:.2f}px, Normalized = {normalized_height:.6f}")
            else:
                print(f"No InsightFace 3D landmarks for Face {i}")
        
        # Add shot type text to image
        shot_text = f"Shot type: {'Wide angle' if is_wide else 'Close-up'} (ratio: {max_face_ratio:.4f})"
        cv2.putText(vis_image, shot_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save or display result
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"diagnostic_{base_name}")
            cv2.imwrite(output_path, vis_image)
            print(f"Saved diagnostic image to {output_path}")
        
        # Display image
        cv2.imshow("Lip Detection Diagnostic", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Lip Detection Diagnostic Tool")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output-dir", help="Directory to save output images")
    
    args = parser.parse_args()
    
    diagnostic = LipDetectionDiagnostic()
    diagnostic.analyze_image(args.image, args.output_dir)

if __name__ == "__main__":
    main() 