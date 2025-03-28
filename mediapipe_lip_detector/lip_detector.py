import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict, deque
import time
import insightface
from insightface.app import FaceAnalysis
import torch

class MediaPipeLipDetector:
    def __init__(self):
        # Initialize InsightFace for robust face detection
        self.app = FaceAnalysis(
            allowed_modules=['detection'], 
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize MediaPipe for accurate lip tracking
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=8,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe lip indices
        self.UPPER_LIP_INDICES = [13, 14, 312]  # Upper lip indices in MediaPipe
        self.LOWER_LIP_INDICES = [17, 16, 15]   # Lower lip indices in MediaPipe
        
        # Parameters for speaking detection
        self.SILENCE_THRESHOLD = 0.0375
        self.MOVEMENT_THRESHOLD = 0.006
        self.SILENCE_DURATION = 2.5    # Increased silence duration
        self.SPEAKING_FRAMES_THRESHOLD = 3
        self.DECAY_RATE = 0.25  # Slower decay
        self.MIN_SPEAKING_DURATION = 1.0  # Minimum speaking duration
        
        # Face tracking system
        self.face_histories = {}
        
        # For consistent face IDs display
        self.next_display_id = 0
        self.id_mapping = {}
        self.active_display_ids = set()
        
        # Frame counter
        self.frame_count = 0
    
    def get_faces_and_lips(self, image):
        # Use InsightFace for robust face detection
        faces = self.app.get(image)
        
        if not faces:
            return []
        
        # Process with MediaPipe for lip landmarks
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_results = self.mp_face_mesh.process(rgb_image)
        
        current_faces = []
        
        # For each InsightFace detection
        for i, face in enumerate(faces):
            bbox = face.bbox  # [x1, y1, x2, y2, score]
            
            # For close-up shots, use MediaPipe if available
            if mp_results and mp_results.multi_face_landmarks:
                # Find closest MediaPipe face
                best_mp_face = None
                best_distance = float('inf')
                
                face_center_x = (bbox[0] + bbox[2]) / 2
                face_center_y = (bbox[1] + bbox[3]) / 2
                
                for j, mp_face in enumerate(mp_results.multi_face_landmarks):
                    # Calculate MediaPipe face center
                    mp_x = np.mean([lm.x for lm in mp_face.landmark]) * image.shape[1]
                    mp_y = np.mean([lm.y for lm in mp_face.landmark]) * image.shape[0]
                    
                    # Calculate distance
                    distance = np.sqrt((face_center_x - mp_x)**2 + (face_center_y - mp_y)**2)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_mp_face = mp_face
                
                # If we found a matching MediaPipe face and it's close enough
                if best_mp_face and best_distance < (bbox[2] - bbox[0]) * 0.5:
                    # Calculate lip height using MediaPipe landmarks
                    lip_height = self.calculate_lip_height_mediapipe(best_mp_face, image.shape)
                    
                    # Store detection
                    current_faces.append({
                        'id': i,
                        'height': lip_height,
                        'bbox': bbox,
                        'mp_landmarks': best_mp_face,
                    })
                    continue
            
            # If MediaPipe failed, skip this face
            # We're only using MediaPipe for lip detection in this simplified version
        
        # Assign display IDs
        for face in current_faces:
            face_id = face['id']
            
            # Assign display ID if this is a new face
            if face_id not in self.id_mapping:
                # Find the lowest available ID
                available_id = 0
                used_ids = set(self.id_mapping.values())
                
                # Find the first available ID starting from 0
                while available_id in used_ids:
                    available_id += 1
                
                self.id_mapping[face_id] = available_id
                self.active_display_ids.add(available_id)
            
            # Store display ID in face data
            face['display_id'] = self.id_mapping[face_id]
        
        return current_faces
    
    def detect_speaking(self, frame):
        self.frame_count += 1
        faces_data = self.get_faces_and_lips(frame)
        current_time = time.time()
        
        # Get current face IDs in this frame
        current_face_ids = {face_data['id'] for face_data in faces_data}
        
        # Only remove histories for faces that haven't been seen in a while
        face_ids_to_remove = []
        for face_id in list(self.face_histories.keys()):
            if face_id not in current_face_ids:
                # If this face hasn't been seen for more than 5 seconds, remove it
                if current_time - self.face_histories[face_id]['last_speaking_time'] > 5.0:
                    face_ids_to_remove.append(face_id)
                    
                    # Find the corresponding ID to remove from mapping
                    if face_id in self.id_mapping:
                        self.active_display_ids.discard(self.id_mapping[face_id])
                        del self.id_mapping[face_id]

        # Remove old face histories
        for face_id in face_ids_to_remove:
            if face_id in self.face_histories:
                del self.face_histories[face_id]
        
        # Process each face in the current frame
        for face_data in faces_data:
            face_id = face_data['id']
            display_id = face_data.get('display_id', face_id)
            
            # Initialize history for this face if needed
            if face_id not in self.face_histories:
                self.face_histories[face_id] = {
                    'last_heights': deque(maxlen=15),
                    'last_speaking_time': current_time,
                    'is_speaking': False,
                    'speaking_frames_count': 0,
                    'display_id': display_id,
                    'speaking_start_time': current_time
                }
            else:
                # Update display ID in history
                self.face_histories[face_id]['display_id'] = display_id
            
            face_history = self.face_histories[face_id]
            
            # Use lip height for speaking detection
            lip_height = face_data.get('height', 0.0)
            face_history['last_heights'].append(lip_height)
            
            if len(face_history['last_heights']) >= 5:
                recent_heights = list(face_history['last_heights'])[-5:]
                variation = np.std(recent_heights)
                
                is_moving = variation > self.MOVEMENT_THRESHOLD
                is_open = lip_height > self.SILENCE_THRESHOLD
                
                if is_moving and is_open:
                    face_history['speaking_frames_count'] += 2
                    if face_history['speaking_frames_count'] >= self.SPEAKING_FRAMES_THRESHOLD:
                        if not face_history['is_speaking']:
                            # Just started speaking - record the time
                            face_history['speaking_start_time'] = current_time
                        face_history['last_speaking_time'] = current_time
                        face_history['is_speaking'] = True
                else:
                    face_history['speaking_frames_count'] = max(0, face_history['speaking_frames_count'] - self.DECAY_RATE)
                    # Only transition to silent if minimum speaking duration has passed
                    if (face_history['is_speaking'] and 
                        current_time - face_history['speaking_start_time'] > self.MIN_SPEAKING_DURATION and
                        current_time - face_history['last_speaking_time'] > self.SILENCE_DURATION):
                        face_history['is_speaking'] = False
            
            # Draw debug visualization
            frame = self.draw_debug(frame, face_data, face_history['is_speaking'], display_id)
        
        # Add face counter and speaking status to top-left corner
        y_offset = 30
        cv2.putText(frame, f"Total faces: {len(faces_data)}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add speaking status for each face
        active_faces = {}
        for face_id, data in self.face_histories.items():
            if face_id in current_face_ids:  # Only show active faces
                display_id = data['display_id']
                active_faces[display_id] = data['is_speaking']

        # Sort by display ID for consistent ordering
        for display_id in sorted(active_faces.keys()):
            y_offset += 30
            is_speaking = active_faces[display_id]
            status = "Speaking" if is_speaking else "Silent"
            color = (0, 255, 0) if is_speaking else (0, 0, 255)
            cv2.putText(frame, f"Face {display_id}: {status}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Clean up ID mapping periodically
        if self.frame_count % 150 == 0:
            # Remove mappings for faces that are no longer tracked
            for face_id in list(self.id_mapping.keys()):
                if face_id not in self.face_histories:
                    if face_id in self.id_mapping:
                        self.active_display_ids.discard(self.id_mapping[face_id])
                        del self.id_mapping[face_id]
        
        return frame, {face_id: data['is_speaking'] 
                      for face_id, data in self.face_histories.items()}
    
    def draw_debug(self, frame, face_data, is_speaking, display_id):
        # Get bounding box
        bbox = face_data['bbox']
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Draw face ID text above the face
        face_id_text = f"Face {display_id}"
        cv2.putText(frame, face_id_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Color based on speaking status
        color = (0, 255, 0) if is_speaking else (0, 0, 255)
        
        # Draw face bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw lip landmarks if MediaPipe landmarks are available
        if face_data['mp_landmarks'] is not None:
            # Draw MediaPipe landmarks
            lip_indices = self.UPPER_LIP_INDICES + self.LOWER_LIP_INDICES
            
            # Make dots larger for better visibility
            dot_size = 2
            
            for idx in lip_indices:
                landmark = face_data['mp_landmarks'].landmark[idx]
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), dot_size, color, -1)
        
        return frame
    
    def calculate_lip_height_mediapipe(self, face_landmarks, image_shape):
        """Calculate normalized lip height using MediaPipe landmarks"""
        h, w = image_shape[:2]
        
        # Get upper and lower lip points
        upper_lip_y = np.mean([face_landmarks.landmark[idx].y for idx in self.UPPER_LIP_INDICES]) * h
        lower_lip_y = np.mean([face_landmarks.landmark[idx].y for idx in self.LOWER_LIP_INDICES]) * h
        
        # Calculate lip distance
        lip_height = abs(upper_lip_y - lower_lip_y)
        
        # Normalize by face height (using nose to chin)
        nose_y = face_landmarks.landmark[1].y * h  # Nose tip
        chin_y = face_landmarks.landmark[152].y * h  # Chin
        face_height = abs(nose_y - chin_y)
        
        return lip_height / face_height if face_height > 0 else 0.0 