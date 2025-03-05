import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict, deque
import time
import insightface
from insightface.app import FaceAnalysis

class LipDetector:
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
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe lip indices
        self.UPPER_LIP_INDICES = [13, 14, 312]  # Upper lip indices in MediaPipe
        self.LOWER_LIP_INDICES = [17, 16, 15]   # Lower lip indices in MediaPipe
        
        # Adjust parameters for more stable detection
        self.SILENCE_THRESHOLD = 0.035
        self.MOVEMENT_THRESHOLD = 0.006
        self.SILENCE_DURATION = 1.2    # Increased silence duration
        self.SPEAKING_FRAMES_THRESHOLD = 3
        
        # Face tracking system
        self.face_histories = defaultdict(lambda: {
            'last_heights': deque(maxlen=15),
            'last_speaking_time': time.time(),
            'is_speaking': False,
            'speaking_frames_count': 0
        })
        
        # Face tracking parameters
        self.face_trackers = {}
        self.next_face_id = 0
        self.face_timeout = 60
        self.iou_threshold = 0.3
    
    def get_faces_and_lips(self, image):
        # Use InsightFace for robust face detection
        faces = self.app.get(image)
        
        if not faces:
            # Update tracking timeouts
            for face_id in list(self.face_trackers.keys()):
                self.face_trackers[face_id]['frames_missing'] += 1
                if self.face_trackers[face_id]['frames_missing'] > self.face_timeout:
                    del self.face_trackers[face_id]
            return []
        
        # Current detections for matching
        current_faces = []
        
        # Process with MediaPipe for accurate lip landmarks
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_results = self.mp_face_mesh.process(rgb_image)
        
        # First, match InsightFace detections with MediaPipe results
        if mp_results.multi_face_landmarks:
            mp_faces = mp_results.multi_face_landmarks
            
            # For each InsightFace detection
            for face in faces:
                bbox = face.bbox  # [x1, y1, x2, y2, score]
                face_center_x = (bbox[0] + bbox[2]) / 2
                face_center_y = (bbox[1] + bbox[3]) / 2
                
                # Find closest MediaPipe face
                best_mp_face = None
                best_distance = float('inf')
                
                for i, mp_face in enumerate(mp_faces):
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
                        'temp_id': len(current_faces),
                        'height': lip_height,
                        'bbox': bbox,
                        'mp_landmarks': best_mp_face
                    })
        
        # If no MediaPipe matches, use InsightFace detections alone
        if not current_faces and faces:
            for face in faces:
                bbox = face.bbox
                # Use a default height since we don't have MediaPipe data
                current_faces.append({
                    'temp_id': len(current_faces),
                    'height': 0.1,  # Default value
                    'bbox': bbox,
                    'mp_landmarks': None
                })
        
        # Match current faces with tracked faces
        matched_faces = []
        unmatched_detections = list(range(len(current_faces)))
        
        # For each tracked face, find best matching detection
        for face_id in list(self.face_trackers.keys()):
            tracker = self.face_trackers[face_id]
            best_match = -1
            best_iou = self.iou_threshold
            
            for i in unmatched_detections:
                iou = self.calculate_iou(tracker['bbox'], current_faces[i]['bbox'])
                if iou > best_iou:
                    best_match = i
                    best_iou = iou
            
            if best_match >= 0:
                # Update tracker with new detection
                tracker['bbox'] = current_faces[best_match]['bbox']
                tracker['frames_missing'] = 0
                
                # Add to matched faces with consistent ID
                face_data = current_faces[best_match]
                face_data['id'] = face_id
                matched_faces.append(face_data)
                
                # Remove from unmatched
                unmatched_detections.remove(best_match)
            else:
                # Face not found in current frame
                tracker['frames_missing'] += 1
                if tracker['frames_missing'] > self.face_timeout:
                    del self.face_trackers[face_id]
        
        # Create new trackers for unmatched detections
        for i in unmatched_detections:
            face_id = self.next_face_id
            self.next_face_id += 1
            
            # Create new tracker
            self.face_trackers[face_id] = {
                'bbox': current_faces[i]['bbox'],
                'frames_missing': 0
            }
            
            # Add to matched faces with new ID
            face_data = current_faces[i]
            face_data['id'] = face_id
            matched_faces.append(face_data)
        
        return matched_faces
    
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
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def draw_debug(self, frame, face_data, is_speaking, face_id):
        # Get bounding box
        bbox = face_data['bbox']
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Draw face ID text above the face
        face_id_text = f"Face {face_id}"
        cv2.putText(frame, face_id_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw lip landmarks if MediaPipe landmarks are available
        if face_data['mp_landmarks'] is not None:
            # Get lip landmark indices (upper and lower lip points)
            lip_indices = self.UPPER_LIP_INDICES + self.LOWER_LIP_INDICES
            
            # Draw dots for lip landmarks
            for idx in lip_indices:
                landmark = face_data['mp_landmarks'].landmark[idx]
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                
                # Draw a small circle at each lip landmark
                # Green for speaking, red for silent
                color = (0, 255, 0) if is_speaking else (0, 0, 255)
                cv2.circle(frame, (x, y), 2, color, -1)
        
        return frame
    
    def detect_speaking(self, frame):
        faces_data = self.get_faces_and_lips(frame)
        current_time = time.time()
        
        # Remove old faces from history
        active_faces = set(face['id'] for face in faces_data)
        for face_id in list(self.face_histories.keys()):
            if face_id not in active_faces:
                if current_time - self.face_histories[face_id]['last_speaking_time'] > 2.0:
                    del self.face_histories[face_id]
        
        # More stable speaking detection with hysteresis
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
                    face_history['speaking_frames_count'] += 2
                    if face_history['speaking_frames_count'] >= self.SPEAKING_FRAMES_THRESHOLD:
                        face_history['last_speaking_time'] = current_time
                        face_history['is_speaking'] = True
                else:
                    face_history['speaking_frames_count'] = max(0, face_history['speaking_frames_count'] - 0.5)
                    if current_time - face_history['last_speaking_time'] > self.SILENCE_DURATION:
                        face_history['is_speaking'] = False
            
            # Draw debug visualization
            frame = self.draw_debug(frame, face_data, face_history['is_speaking'], face_id)
        
        return frame, {face_id: data['is_speaking'] 
                      for face_id, data in self.face_histories.items()} 