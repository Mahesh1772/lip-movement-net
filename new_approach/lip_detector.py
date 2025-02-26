import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class LipDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Parameters for speaking detection
        self.SILENCE_THRESHOLD = 0.038    # Normalized lip distance threshold
        self.MOVEMENT_THRESHOLD = 0.004   # Minimum change to consider movement
        self.SILENCE_DURATION = 1.0       # Seconds of closed lips to consider silent
        self.last_speaking_time = time.time()
        self.is_speaking = False
        
        # Buffer for movement detection
        self.distance_history = deque(maxlen=5)  # Reduced buffer size for quicker response
        self.last_heights = deque(maxlen=10)     # Buffer for variation detection
        
    def get_lip_height(self, image):
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get face landmarks
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None, None
            
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get lip landmarks (upper and lower lip)
        upper_lip = np.array([face_landmarks.landmark[13].y, 
                            face_landmarks.landmark[14].y])
        lower_lip = np.array([face_landmarks.landmark[17].y,
                            face_landmarks.landmark[16].y])
        
        # Calculate lip height
        lip_height = np.mean(np.abs(upper_lip - lower_lip))
        
        # Normalize by face height
        face_height = abs(face_landmarks.landmark[152].y - face_landmarks.landmark[10].y)
        normalized_height = lip_height / face_height
        
        return normalized_height, face_landmarks
        
    def detect_speaking(self, frame):
        lip_height, landmarks = self.get_lip_height(frame)
        
        if lip_height is None:
            return frame, False
            
        # Add to history
        self.distance_history.append(lip_height)
        self.last_heights.append(lip_height)
        
        current_time = time.time()
        
        # Detect movement by checking variation in recent heights
        if len(self.last_heights) >= 3:
            variation = np.std(list(self.last_heights)[-3:])
            height_change = variation > self.MOVEMENT_THRESHOLD
            
            # Update speaking status based on movement and height
            if height_change and lip_height > self.SILENCE_THRESHOLD:
                self.last_speaking_time = current_time
                self.is_speaking = True
            elif current_time - self.last_speaking_time > self.SILENCE_DURATION:
                self.is_speaking = False
            
        # Draw landmarks and status
        frame = self.draw_debug(frame, landmarks, lip_height)
        
        return frame, self.is_speaking
        
    def draw_debug(self, frame, landmarks, height):
        h, w, _ = frame.shape
        
        # Draw lip landmarks
        for idx in [13, 14, 17, 16]:  # Lip landmark indices
            pos = landmarks.landmark[idx]
            cv2.circle(frame, 
                      (int(pos.x * w), int(pos.y * h)), 
                      2, (0, 255, 0), -1)
        
        # Display info
        cv2.putText(frame, f"Height: {height:.3f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if len(self.last_heights) >= 3:
            variation = np.std(list(self.last_heights)[-3:])
            cv2.putText(frame, f"Movement: {variation:.4f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        status_color = (0, 255, 0) if self.is_speaking else (0, 0, 255)
        status_text = "Speaking" if self.is_speaking else "Silent"
        cv2.putText(frame, status_text, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                   
        return frame 