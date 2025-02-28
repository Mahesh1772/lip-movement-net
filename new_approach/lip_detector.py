import cv2
import numpy as np
from collections import defaultdict, deque
import time
import insightface
from insightface.app import FaceAnalysis

class LipDetector:
    def __init__(self):
        # Initialize InsightFace
        self.app = FaceAnalysis(
            allowed_modules=['detection', 'landmark_3d_68'], 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Adjust parameters for more stable detection
        self.SILENCE_THRESHOLD = 0.035
        self.MOVEMENT_THRESHOLD = 0.006
        self.SILENCE_DURATION = 0.8    # Increased from 0.5 to 0.8 seconds
        self.SPEAKING_FRAMES_THRESHOLD = 4  # New: require N consecutive speaking frames
        
        # Define lip landmarks for 68-point model
        self.UPPER_LIP_INDICES = [50, 51, 52]  # Upper lip indices
        self.LOWER_LIP_INDICES = [58, 57, 56]  # Lower lip indices
        
        # Modified face tracking system
        self.face_histories = defaultdict(lambda: {
            'distance_history': deque(maxlen=5),
            'last_heights': deque(maxlen=10),
            'last_speaking_time': time.time(),
            'is_speaking': False,
            'speaking_frames_count': 0  # New: count consecutive speaking frames
        })
    
    def get_lip_heights(self, image):
        # Get face analysis from InsightFace
        faces = self.app.get(image)
        
        if not faces:
            return []
            
        face_data = []
        
        for face_idx, face in enumerate(faces):
            landmarks = face.landmark_3d_68
            if landmarks is None:
                continue
                
            # Calculate normalized height using 68-point landmarks
            lip_height = self.calculate_lip_height(landmarks)
            
            face_data.append({
                'id': face_idx,
                'height': lip_height,
                'landmarks': landmarks,
                'bbox': face.bbox
            })
        
        return face_data
    
    def calculate_lip_height(self, landmarks):
        """Calculate normalized lip height using 68-point landmarks"""
        # Get upper and lower lip points
        upper_lip_y = np.mean(landmarks[self.UPPER_LIP_INDICES][:, 1])
        lower_lip_y = np.mean(landmarks[self.LOWER_LIP_INDICES][:, 1])
        
        # Calculate lip distance
        lip_height = abs(upper_lip_y - lower_lip_y)
        
        # Normalize by face height (using nose bridge to chin)
        face_height = abs(landmarks[27][1] - landmarks[8][1])  # Nose bridge to chin
        
        return lip_height / face_height if face_height > 0 else 0.0
    
    def draw_debug(self, frame, landmarks, height, is_speaking, face_id):
        h, w, _ = frame.shape
        
        # Calculate text position (keeping only the text, removing dots)
        text_y = int(landmarks[27][1]) - 20  # Above nose bridge
        text_x = int(landmarks[27][0])
        
        # Display info
        status_color = (0, 255, 0) if is_speaking else (0, 0, 255)
        status_text = f"Face {face_id+1}: {'Speaking' if is_speaking else 'Silent'}"
        cv2.putText(frame, f"{status_text} ({height:.3f})", 
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        return frame
    
    def detect_speaking(self, frame):
        faces_data = self.get_lip_heights(frame)
        current_time = time.time()
        
        # Remove old faces
        active_faces = set(face['id'] for face in faces_data)
        for face_id in list(self.face_histories.keys()):
            if face_id not in active_faces:
                if current_time - self.face_histories[face_id]['last_speaking_time'] > 2.0:  # Reduced timeout
                    del self.face_histories[face_id]
        
        # More stable speaking detection
        for face_data in faces_data:
            face_id = face_data['id']
            lip_height = face_data['height']
            face_history = self.face_histories[face_id]
            
            face_history['last_heights'].append(lip_height)
            if len(face_history['last_heights']) >= 5:
                recent_heights = list(face_history['last_heights'])[-5:]
                variation = np.std(recent_heights)
                
                is_moving = variation > self.MOVEMENT_THRESHOLD
                is_open = lip_height > self.SILENCE_THRESHOLD
                
                if is_moving and is_open:
                    face_history['speaking_frames_count'] += 1
                    if face_history['speaking_frames_count'] >= self.SPEAKING_FRAMES_THRESHOLD:
                        face_history['last_speaking_time'] = current_time
                        face_history['is_speaking'] = True
                else:
                    face_history['speaking_frames_count'] = 0
                    if current_time - face_history['last_speaking_time'] > self.SILENCE_DURATION:
                        face_history['is_speaking'] = False
            
            # Draw debug visualization
            frame = self.draw_debug(frame, face_data['landmarks'], 
                                 lip_height, face_history['is_speaking'], face_id)
        
        return frame, {face_id: data['is_speaking'] 
                      for face_id, data in self.face_histories.items()} 