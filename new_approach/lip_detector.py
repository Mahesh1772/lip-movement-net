import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict, deque
import time
import insightface
from insightface.app import FaceAnalysis
import sys
import os
import torch

# Import directly from the specific location
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch

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
        self.SILENCE_THRESHOLD = 0.0375
        self.MOVEMENT_THRESHOLD = 0.006
        self.SILENCE_DURATION = 1.5    # Increased silence duration
        self.SPEAKING_FRAMES_THRESHOLD = 3
        
        # Face tracking system
        self.face_histories = defaultdict(lambda: {
            'last_heights': deque(maxlen=15),
            'last_speaking_time': time.time(),
            'is_speaking': False,
            'speaking_frames_count': 0
        })
        
        # Initialize ByteTracker
        class Args:
            def __init__(self):
                self.track_thresh = 0.5
                self.track_buffer = 30
                self.match_thresh = 0.8
                self.frame_rate = 30
                self.mot20 = False  # Add this missing attribute

        self.tracker = BYTETracker(Args())
        
        self.frame_id = 0  # Frame counter for ByteTracker
    
    def get_faces_and_lips(self, image):
        # Use InsightFace for robust face detection
        faces = self.app.get(image)
        
        if not faces:
            # Update ByteTracker with empty detections
            self.frame_id += 1
            empty_tensor = torch.zeros((0, 6))
            self.tracker.update(
                empty_tensor,
                [image.shape[0], image.shape[1]],
                [image.shape[0], image.shape[1]]
            )
            return []
        
        # Determine if this is a wide angle shot
        is_wide_shot = self.is_wide_angle_shot(image, faces)

        
        # Current detections for ByteTracker
        detections = []
        scores = []
        current_faces = []
        
        # For wide shots, we need to be more aggressive with lip detection
        if is_wide_shot:
            # Increase MediaPipe confidence thresholds for wide shots
            self.mp_face_mesh.min_detection_confidence = 0.3
            self.mp_face_mesh.min_tracking_confidence = 0.3
        else:
            # Reset to normal values for close-ups
            self.mp_face_mesh.min_detection_confidence = 0.5
            self.mp_face_mesh.min_tracking_confidence = 0.5
        
        # Process with MediaPipe for lip landmarks (always)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_results = self.mp_face_mesh.process(rgb_image)
        
        if mp_results.multi_face_landmarks:
            mp_faces = mp_results.multi_face_landmarks
            
            # For each InsightFace detection
            for face in faces:
                bbox = face.bbox  # [x1, y1, x2, y2, score]
                face_center_x = (bbox[0] + bbox[2]) / 2
                face_center_y = (bbox[1] + bbox[3]) / 2
                
                # Add to ByteTracker detections
                detections.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                scores.append(face.det_score)
                
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
                    # Calculate lip height using appropriate method based on shot type
                    if is_wide_shot and hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
                        # Use InsightFace landmarks for lip height in wide shots
                        lip_height = self.calculate_lip_height_insightface(face.landmark_3d_68)
                    else:
                        # Use MediaPipe landmarks for lip height
                        lip_height = self.calculate_lip_height_mediapipe(best_mp_face, image.shape)
                    
                    # Store detection
                    current_faces.append({
                        'temp_id': len(current_faces),
                        'height': lip_height,
                        'bbox': bbox,
                        'mp_landmarks': best_mp_face,
                        'is_wide_shot': is_wide_shot
                    })
        
        # If no MediaPipe matches, use InsightFace detections alone
        if not current_faces and faces:
            for i, face in enumerate(faces):
                bbox = face.bbox
                # Use a default height since we don't have MediaPipe data
                current_faces.append({
                    'temp_id': i,
                    'height': 0.1,  # Default value
                    'bbox': bbox,
                    'mp_landmarks': None,
                    'is_wide_shot': is_wide_shot
                })
        
        # Update ByteTracker with current detections
        self.frame_id += 1
        
        # Convert detections to numpy array for ByteTracker
        if detections:
            # Format: [x1, y1, x2, y2, obj_conf, class_conf]
            formatted_detections = []
            for i, box in enumerate(detections):
                # Add confidence scores (obj_conf and class_conf)
                formatted_box = list(box) + [scores[i], 1.0]  # Add object confidence and class confidence
                formatted_detections.append(formatted_box)
            
            detections_np = np.array(formatted_detections)
            
            # Convert NumPy arrays to PyTorch tensors
            detections_tensor = torch.from_numpy(detections_np)
            
            # Update tracker
            online_targets = self.tracker.update(
                detections_tensor,
                [image.shape[0], image.shape[1]],  # Image dimensions (height, width)
                [image.shape[0], image.shape[1]]   # Image size (same as dimensions in this case)
            )
            
            # Match tracked objects with current faces
            if online_targets and current_faces:
                # Get bounding boxes from tracked objects
                track_boxes = np.array([[t.tlbr[0], t.tlbr[1], t.tlbr[2], t.tlbr[3]] for t in online_targets])
                
                # Get bounding boxes from current faces
                face_boxes = np.array([[f['bbox'][0], f['bbox'][1], f['bbox'][2], f['bbox'][3]] for f in current_faces])
                
                # Calculate IoU between tracked boxes and face boxes
                iou_matrix = box_iou_batch(track_boxes, face_boxes)
                
                # Match faces with tracks
                for i, track in enumerate(online_targets):
                    if i < len(iou_matrix):
                        best_match = np.argmax(iou_matrix[i])
                        if best_match < len(current_faces) and iou_matrix[i][best_match] > 0.5:
                            # Assign track ID to face
                            current_faces[best_match]['id'] = track.track_id
        
        # Ensure all faces have an ID (fallback to temp_id if not matched)
        for face in current_faces:
            if 'id' not in face:
                face['id'] = face['temp_id']
        
        return current_faces
    
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
        color = (0, 255, 0) if is_speaking else (0, 0, 255)
        
        if face_data['mp_landmarks'] is not None:
            # Draw MediaPipe landmarks for all shots
            lip_indices = self.UPPER_LIP_INDICES + self.LOWER_LIP_INDICES
            
            # Make dots larger in wide shots for better visibility
            dot_size = 4 if face_data.get('is_wide_shot', False) else 2
            
            for idx in lip_indices:
                landmark = face_data['mp_landmarks'].landmark[idx]
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), dot_size, color, -1)
        
        return frame
    
    def detect_speaking(self, frame):
        faces_data = self.get_faces_and_lips(frame)
        current_time = time.time()
        
        # Get current face IDs in this frame
        current_face_ids = {face_data['id'] for face_data in faces_data}
        
        # Only remove histories for faces that haven't been seen in a while
        face_ids_to_remove = []
        for face_id in self.face_histories:
            if face_id not in current_face_ids:
                # If this face hasn't been seen for more than 5 seconds, remove it
                if current_time - self.face_histories[face_id]['last_speaking_time'] > 5.0:
                    face_ids_to_remove.append(face_id)
        
        # Remove old face histories
        for face_id in face_ids_to_remove:
            del self.face_histories[face_id]
        
        # Process each face in the current frame
        for face_data in faces_data:
            face_id = face_data['id']
            lip_height = face_data['height']
            is_wide_shot = face_data.get('is_wide_shot', False)
            
            # Use different thresholds based on shot type
            if is_wide_shot:
                # Much more sensitive thresholds for wide shots
                movement_threshold = self.MOVEMENT_THRESHOLD * 0.25  # 4x more sensitive
                silence_threshold = self.SILENCE_THRESHOLD * 0.6     # Lower threshold
            else:
                movement_threshold = self.MOVEMENT_THRESHOLD
                silence_threshold = self.SILENCE_THRESHOLD
            
            # Initialize history for this face if needed
            if face_id not in self.face_histories:
                self.face_histories[face_id] = {
                    'last_heights': deque(maxlen=15),
                    'last_speaking_time': current_time,
                    'is_speaking': False,
                    'speaking_frames_count': 0
                }
            
            face_history = self.face_histories[face_id]
            face_history['last_heights'].append(lip_height)
            
            if len(face_history['last_heights']) >= 5:
                recent_heights = list(face_history['last_heights'])[-5:]
                variation = np.std(recent_heights)
                
                is_moving = variation > movement_threshold
                is_open = lip_height > silence_threshold
                
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
    
    def is_wide_angle_shot(self, image, faces):
        """
        Determine if the current frame is a wide angle shot based solely on
        face size relative to the frame
        """
        if not faces:
            print(f"Frame {self.frame_id}: No faces detected")
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
        
        # Use face size as the only criterion - adjust threshold as needed
        is_wide = max_face_ratio < 0.02
        
        return is_wide

    def calculate_lip_height_insightface(self, landmarks):
        """Calculate normalized lip height using InsightFace landmarks"""
        # InsightFace landmark indices for upper and lower lips
        # These may need adjustment based on InsightFace's landmark mapping
        upper_lip_indices = [62, 63, 64]  # Upper lip indices in InsightFace
        lower_lip_indices = [66, 67, 68]  # Lower lip indices in InsightFace
        
        # Get upper and lower lip points
        upper_lip_y = np.mean([landmarks[idx][1] for idx in upper_lip_indices])
        lower_lip_y = np.mean([landmarks[idx][1] for idx in lower_lip_indices])
        
        # Calculate lip distance
        lip_height = abs(upper_lip_y - lower_lip_y)
        
        # Normalize by face height (using nose to chin)
        nose_y = landmarks[30][1]  # Nose tip
        chin_y = landmarks[8][1]   # Chin
        face_height = abs(nose_y - chin_y)
        
        return lip_height / face_height if face_height > 0 else 0.0 