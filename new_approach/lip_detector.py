import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict, deque
import time

class LipDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,  # Increased max faces
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Parameters for speaking detection
        self.SILENCE_THRESHOLD = 0.038    # Normalized lip distance threshold
        self.MOVEMENT_THRESHOLD = 0.004   # Minimum change to consider movement
        self.SILENCE_DURATION = 1.0       # Seconds of closed lips to consider silent
        
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
        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
            # Get lip landmarks
            upper_lip = np.array([face_landmarks.landmark[13].y, 
                                face_landmarks.landmark[14].y])
            lower_lip = np.array([face_landmarks.landmark[17].y,
                                face_landmarks.landmark[16].y])
            
            # Calculate normalized height
            lip_height = np.mean(np.abs(upper_lip - lower_lip))
            face_height = abs(face_landmarks.landmark[152].y - face_landmarks.landmark[10].y)
            normalized_height = lip_height / face_height
            
            face_data.append({
                'id': face_idx,
                'height': normalized_height,
                'landmarks': face_landmarks
            })
            
        return face_data
        
    def detect_speaking(self, frame):
        faces_data = self.get_lip_heights(frame)
        current_time = time.time()
        
        # Remove old faces that haven't been seen recently
        active_faces = set(face['id'] for face in faces_data)
        for face_id in list(self.face_histories.keys()):
            if face_id not in active_faces:
                if current_time - self.face_histories[face_id]['last_speaking_time'] > 5.0:
                    del self.face_histories[face_id]
        
        # Process each detected face
        for face_data in faces_data:
            face_id = face_data['id']
            lip_height = face_data['height']
            face_history = self.face_histories[face_id]
            
            # Update histories
            face_history['distance_history'].append(lip_height)
            face_history['last_heights'].append(lip_height)
            
            # Detect movement
            if len(face_history['last_heights']) >= 3:
                variation = np.std(list(face_history['last_heights'])[-3:])
                height_change = variation > self.MOVEMENT_THRESHOLD
                
                # Update speaking status
                if height_change and lip_height > self.SILENCE_THRESHOLD:
                    face_history['last_speaking_time'] = current_time
                    face_history['is_speaking'] = True
                elif current_time - face_history['last_speaking_time'] > self.SILENCE_DURATION:
                    face_history['is_speaking'] = False
            
            # Draw debug for this face
            frame = self.draw_debug(frame, face_data['landmarks'], 
                                  lip_height, face_history['is_speaking'], face_id)
        
        return frame, {face_id: data['is_speaking'] 
                      for face_id, data in self.face_histories.items()}
        
    def draw_debug(self, frame, landmarks, height, is_speaking, face_id):
        h, w, _ = frame.shape
        
        # Calculate text position based on face position
        nose_tip = landmarks.landmark[4]
        text_x = int(nose_tip.x * w)
        text_y = int(nose_tip.y * h) - 20
        
        # Draw lip landmarks
        for idx in [13, 14, 17, 16]:
            pos = landmarks.landmark[idx]
            cv2.circle(frame, 
                      (int(pos.x * w), int(pos.y * h)), 
                      2, (0, 255, 0), -1)
        
        # Display info near each face
        status_color = (0, 255, 0) if is_speaking else (0, 0, 255)
        status_text = f"Face {face_id+1}: {'Speaking' if is_speaking else 'Silent'}"
        cv2.putText(frame, f"{status_text} ({height:.3f})", 
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                   
        return frame 