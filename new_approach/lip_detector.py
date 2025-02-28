import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict, deque
import time
import tensorflow as tf  # Add TensorFlow import

class LipDetector:
    def __init__(self):
        # Enable GPU growth to prevent TF from taking all memory
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU acceleration enabled!")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        
        # Initialize MediaPipe Face Mesh with lower thresholds
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,          # Keep max faces at 4
            refine_landmarks=True,
            min_detection_confidence=0.5,  # Lower threshold for detection
            min_tracking_confidence=0.5    # Lower threshold for tracking
        )
        
        # Adjust speaking detection parameters
        self.SILENCE_THRESHOLD = 0.035    # Lower - easier to detect open mouth
        self.MOVEMENT_THRESHOLD = 0.004   # Lower - more sensitive to movement
        self.SILENCE_DURATION = 0.5       # Keep quick transition to silence
        
        # Define lip landmarks
        self.UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
        self.LOWER_LIP_INDICES = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        
        # Track multiple faces
        self.face_histories = defaultdict(lambda: {
            'distance_history': deque(maxlen=5),
            'last_heights': deque(maxlen=10),
            'last_speaking_time': time.time(),
            'is_speaking': False
        })
        
    def get_lip_heights(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return []
            
        face_data = []
        h, w = image.shape[:2]
        
        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
            # Calculate normalized height
            lip_height = self.calculate_lip_height(face_landmarks)
            
            face_data.append({
                'id': face_idx,
                'height': lip_height,
                'landmarks': face_landmarks
            })
        
        return face_data
        
    def detect_speaking(self, frame):
        faces_data = self.get_lip_heights(frame)
        current_time = time.time()
        
        # Remove old faces
        active_faces = set(face['id'] for face in faces_data)
        for face_id in list(self.face_histories.keys()):
            if face_id not in active_faces:
                if current_time - self.face_histories[face_id]['last_speaking_time'] > 2.0:  # Reduced timeout
                    del self.face_histories[face_id]
        
        # Process each detected face
        for face_data in faces_data:
            face_id = face_data['id']
            lip_height = face_data['height']
            face_history = self.face_histories[face_id]
            
            # Require more consistent movement for speaking detection
            face_history['last_heights'].append(lip_height)
            if len(face_history['last_heights']) >= 5:  # Increased window size
                recent_heights = list(face_history['last_heights'])[-5:]
                variation = np.std(recent_heights)
                
                # More strict speaking detection
                is_moving = variation > self.MOVEMENT_THRESHOLD
                is_open = lip_height > self.SILENCE_THRESHOLD
                
                if is_moving and is_open:
                    face_history['last_speaking_time'] = current_time
                    face_history['is_speaking'] = True
                elif current_time - face_history['last_speaking_time'] > self.SILENCE_DURATION:
                    face_history['is_speaking'] = False
            
            # Draw debug visualization
            frame = self.draw_debug(frame, face_data['landmarks'], 
                                 lip_height, face_history['is_speaking'], face_id)
        
        return frame, {face_id: data['is_speaking'] 
                      for face_id, data in self.face_histories.items()}
        
    def draw_debug(self, frame, landmarks, height, is_speaking, face_id):
        h, w, _ = frame.shape
        
        # Draw ALL lip landmarks more visibly
        for idx in self.UPPER_LIP_INDICES + self.LOWER_LIP_INDICES:
            pos = landmarks.landmark[idx]
            x, y = int(pos.x * w), int(pos.y * h)
            # Draw larger circles in bright color
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Yellow dots
            
        # Calculate text position based on nose position
        nose_tip = landmarks.landmark[4]
        text_x = int(nose_tip.x * w)
        text_y = int(nose_tip.y * h) - 20
        
        # Display info with height value
        status_color = (0, 255, 0) if is_speaking else (0, 0, 255)
        status_text = f"Face {face_id+1}: {'Speaking' if is_speaking else 'Silent'}"
        cv2.putText(frame, f"{status_text} ({height:.3f})", 
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        return frame 

    def calculate_lip_height(self, landmarks):
        """Calculate normalized lip height using facial landmarks"""
        # Get upper and lower lip points
        upper_lip_y = np.mean([landmarks.landmark[i].y for i in self.UPPER_LIP_INDICES])
        lower_lip_y = np.mean([landmarks.landmark[i].y for i in self.LOWER_LIP_INDICES])
        
        # Calculate lip distance
        lip_height = abs(upper_lip_y - lower_lip_y)
        
        # Normalize by face height (using points 10 and 152 for face height)
        face_height = abs(landmarks.landmark[10].y - landmarks.landmark[152].y)
        
        # Return normalized height
        return lip_height / face_height if face_height > 0 else 0.0 