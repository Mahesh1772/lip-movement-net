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

class EnhancedLipDetector:
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
            max_num_faces=8,  # Increased to handle more faces
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
            'speaking_frames_count': 0,
            'display_id': None  # For consistent display IDs
        })
        
        # Initialize ByteTracker
        class Args:
            def __init__(self):
                self.track_thresh = 0.5
                self.track_buffer = 30
                self.match_thresh = 0.8
                self.frame_rate = 30
                self.mot20 = False

        self.tracker = BYTETracker(Args())
        
        self.frame_id = 0  # Frame counter for ByteTracker
        
        # For wide shots, use a simpler approach based on mouth region intensity changes
        self.face_mouth_regions = {}  # Store mouth regions for each face
        
        # For consistent face IDs display
        self.next_display_id = 0
        self.id_mapping = {}  # Maps ByteTrack IDs to display IDs
        self.active_display_ids = set()  # Track currently active display IDs
        self.max_id_seen = 0  # Track the highest ID we've assigned
    
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
        
        # Process with MediaPipe for lip landmarks
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_results = self.mp_face_mesh.process(rgb_image)
        
        # For each InsightFace detection
        for i, face in enumerate(faces):
            bbox = face.bbox  # [x1, y1, x2, y2, score]
            
            # Add to ByteTracker detections
            detections.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            scores.append(face.det_score)
            
            # For wide shots, use a different approach since MediaPipe often fails
            if is_wide_shot:
                # Extract mouth region based on face bbox
                x1, y1, x2, y2 = map(int, bbox[:4])
                face_width = x2 - x1
                face_height = y2 - y1
                
                # Estimate mouth region (lower third of face)
                mouth_x1 = x1 + int(face_width * 0.25)
                mouth_y1 = y1 + int(face_height * 0.6)
                mouth_x2 = x2 - int(face_width * 0.25)
                mouth_y2 = y2 - int(face_height * 0.1)
                
                # Ensure coordinates are within image bounds
                mouth_x1 = max(0, mouth_x1)
                mouth_y1 = max(0, mouth_y1)
                mouth_x2 = min(image.shape[1], mouth_x2)
                mouth_y2 = min(image.shape[0], mouth_y2)
                
                # Extract mouth region
                mouth_region = image[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
                
                # Store face with estimated mouth region
                current_faces.append({
                    'temp_id': i,
                    'bbox': bbox,
                    'mp_landmarks': None,
                    'is_wide_shot': True,
                    'mouth_region': {
                        'x1': mouth_x1, 'y1': mouth_y1, 
                        'x2': mouth_x2, 'y2': mouth_y2
                    },
                    'height': 0.1  # Default value
                })
                continue
            
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
                        'temp_id': i,
                        'height': lip_height,
                        'bbox': bbox,
                        'mp_landmarks': best_mp_face,
                        'is_wide_shot': False
                    })
                    continue
            
            # If MediaPipe failed or not available, use fallback approach
            x1, y1, x2, y2 = map(int, bbox[:4])
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Estimate mouth region
            mouth_x1 = x1 + int(face_width * 0.25)
            mouth_y1 = y1 + int(face_height * 0.6)
            mouth_x2 = x2 - int(face_width * 0.25)
            mouth_y2 = y2 - int(face_height * 0.1)
            
            # Ensure coordinates are within image bounds
            mouth_x1 = max(0, mouth_x1)
            mouth_y1 = max(0, mouth_y1)
            mouth_x2 = min(image.shape[1], mouth_x2)
            mouth_y2 = min(image.shape[0], mouth_y2)
            
            current_faces.append({
                'temp_id': i,
                'bbox': bbox,
                'mp_landmarks': None,
                'is_wide_shot': is_wide_shot,
                'mouth_region': {
                    'x1': mouth_x1, 'y1': mouth_y1, 
                    'x2': mouth_x2, 'y2': mouth_y2
                },
                'height': 0.1  # Default value
            })
        
        # Update ByteTracker with current detections
        self.frame_id += 1
        
        # Convert detections to numpy array for ByteTracker
        if detections:
            # Format: [x1, y1, x2, y2, obj_conf, class_conf]
            formatted_detections = []
            for i, box in enumerate(detections):
                # Add confidence scores (obj_conf and class_conf)
                formatted_box = list(box) + [scores[i], 1.0]
                formatted_detections.append(formatted_box)
            
            detections_np = np.array(formatted_detections)
            
            # Convert NumPy arrays to PyTorch tensors
            detections_tensor = torch.from_numpy(detections_np)
            
            # Update tracker
            online_targets = self.tracker.update(
                detections_tensor,
                [image.shape[0], image.shape[1]],
                [image.shape[0], image.shape[1]]
            )
            
            # Match faces with tracks
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
                            
                            # Assign display ID if this is a new track
                            if track.track_id not in self.id_mapping:
                                # Find the lowest available ID
                                available_id = 0
                                used_ids = set(self.id_mapping.values())
                                
                                # Find the first available ID starting from 0
                                while available_id in used_ids:
                                    available_id += 1
                                
                                self.id_mapping[track.track_id] = available_id
                                self.max_id_seen = max(self.max_id_seen, available_id)
                                self.active_display_ids.add(available_id)
                            
                            # Store display ID in face data
                            current_faces[best_match]['display_id'] = self.id_mapping[track.track_id]
                            self.active_display_ids.add(self.id_mapping[track.track_id])
        
        # Ensure all faces have an ID (fallback to temp_id if not matched)
        for face in current_faces:
            if 'id' not in face:
                face['id'] = face['temp_id']
                
                # Assign a temporary display ID
                face['display_id'] = 999 + face['temp_id']  # Use high numbers for temporary IDs
        
        return current_faces
    
    def detect_speaking(self, frame):
        faces_data = self.get_faces_and_lips(frame)
        current_time = time.time()
        
        # Get current face IDs in this frame
        current_face_ids = {face_data['id'] for face_data in faces_data}
        
        # Only remove histories for faces that haven't been seen in a while
        face_ids_to_remove = []
        track_ids_to_remove = []
        for face_id in self.face_histories:
            if face_id not in current_face_ids:
                # If this face hasn't been seen for more than 5 seconds, remove it
                if current_time - self.face_histories[face_id]['last_speaking_time'] > 5.0:
                    face_ids_to_remove.append(face_id)
                    
                    # Find the corresponding track ID to remove from mapping
                    for track_id, display_id in self.id_mapping.items():
                        if display_id == self.face_histories[face_id]['display_id']:
                            track_ids_to_remove.append(track_id)
                            self.active_display_ids.discard(display_id)
                            break

        # Remove old face histories
        for face_id in face_ids_to_remove:
            del self.face_histories[face_id]
            if face_id in self.face_mouth_regions:
                del self.face_mouth_regions[face_id]

        # Remove old track ID mappings
        for track_id in track_ids_to_remove:
            if track_id in self.id_mapping:
                del self.id_mapping[track_id]
        
        # Process each face in the current frame
        for face_data in faces_data:
            face_id = face_data['id']
            display_id = face_data.get('display_id', face_id)  # Use display_id if available
            is_wide_shot = face_data.get('is_wide_shot', False)
            
            # Initialize history for this face if needed
            if face_id not in self.face_histories:
                self.face_histories[face_id] = {
                    'last_heights': deque(maxlen=15),
                    'last_speaking_time': current_time,
                    'is_speaking': False,
                    'speaking_frames_count': 0,
                    'mouth_diff_history': deque(maxlen=10),
                    'display_id': display_id
                }
            else:
                # Update display ID in history
                self.face_histories[face_id]['display_id'] = display_id
            
            face_history = self.face_histories[face_id]
            
            # Different processing for wide shots vs close-ups
            if is_wide_shot and 'mouth_region' in face_data:
                # For wide shots, use mouth region intensity changes
                mouth_region = face_data['mouth_region']
                mouth_img = frame[mouth_region['y1']:mouth_region['y2'], 
                                  mouth_region['x1']:mouth_region['x2']]
                
                # Store current mouth region
                if face_id not in self.face_mouth_regions:
                    self.face_mouth_regions[face_id] = mouth_img.copy()
                    face_history['mouth_diff_history'].append(0)
                else:
                    # Compare with previous mouth region
                    prev_mouth = self.face_mouth_regions[face_id]
                    
                    # Resize if dimensions don't match
                    if prev_mouth.shape != mouth_img.shape and mouth_img.size > 0 and prev_mouth.size > 0:
                        prev_mouth = cv2.resize(prev_mouth, (mouth_img.shape[1], mouth_img.shape[0]))
                    
                    # Calculate difference if both regions are valid
                    if mouth_img.size > 0 and prev_mouth.size > 0 and mouth_img.shape == prev_mouth.shape:
                        # Convert to grayscale
                        if len(mouth_img.shape) > 2:
                            mouth_gray = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)
                            prev_mouth_gray = cv2.cvtColor(prev_mouth, cv2.COLOR_BGR2GRAY)
                        else:
                            mouth_gray = mouth_img
                            prev_mouth_gray = prev_mouth
                        
                        # Calculate absolute difference
                        diff = cv2.absdiff(mouth_gray, prev_mouth_gray)
                        mean_diff = np.mean(diff)
                        
                        # Store difference
                        face_history['mouth_diff_history'].append(mean_diff)
                        
                        # Update stored mouth region
                        self.face_mouth_regions[face_id] = mouth_img.copy()
                    else:
                        face_history['mouth_diff_history'].append(0)
                
                # Determine if speaking based on mouth region changes
                if len(face_history['mouth_diff_history']) >= 5:
                    # Calculate recent movement
                    recent_diffs = list(face_history['mouth_diff_history'])[-5:]
                    mean_diff = np.mean(recent_diffs)
                    std_diff = np.std(recent_diffs)
                    
                    # Thresholds for wide shots
                    movement_threshold = 2.0  # Adjust based on testing
                    
                    # Determine if speaking
                    is_moving = mean_diff > movement_threshold or std_diff > movement_threshold/2
                    
                    if is_moving:
                        face_history['speaking_frames_count'] += 1
                        if face_history['speaking_frames_count'] >= self.SPEAKING_FRAMES_THRESHOLD:
                            face_history['last_speaking_time'] = current_time
                            face_history['is_speaking'] = True
                    else:
                        face_history['speaking_frames_count'] = max(0, face_history['speaking_frames_count'] - 0.5)
                        if current_time - face_history['last_speaking_time'] > self.SILENCE_DURATION:
                            face_history['is_speaking'] = False
            else:
                # For close-ups, use lip height
                lip_height = face_data.get('height', 0.1)
                face_history['last_heights'].append(lip_height)
                
                if len(face_history['last_heights']) >= 5:
                    recent_heights = list(face_history['last_heights'])[-5:]
                    variation = np.std(recent_heights)
                    
                    
                    # Standard thresholds for close-ups
                    movement_threshold = self.MOVEMENT_THRESHOLD
                    silence_threshold = self.SILENCE_THRESHOLD
                    
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
        
        # Periodically reset IDs if we have fewer faces than max_id_seen
        if self.frame_id % 150 == 0 and len(faces_data) < self.max_id_seen:
            # Get current active face IDs
            active_track_ids = set()
            for face in faces_data:
                if 'id' in face:
                    active_track_ids.add(face['id'])
            
            # Get active display IDs
            active_display_ids = set()
            for track_id, display_id in list(self.id_mapping.items()):
                if track_id in active_track_ids:
                    active_display_ids.add(display_id)
            
            # If we have gaps in our IDs, reassign them
            if len(active_display_ids) < max(active_display_ids) + 1:
                # Create new mapping
                new_mapping = {}
                new_id = 0
                
                # Assign new sequential IDs
                for track_id in active_track_ids:
                    if track_id in self.id_mapping:
                        new_mapping[track_id] = new_id
                        new_id += 1
                
                # Update mapping
                self.id_mapping = new_mapping
                self.max_id_seen = new_id - 1
                self.active_display_ids = set(range(new_id))
                
                # Update face display IDs
                for face in faces_data:
                    if 'id' in face and face['id'] in self.id_mapping:
                        face['display_id'] = self.id_mapping[face['id']]
        
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
        
        # Draw lip landmarks if MediaPipe landmarks are available
        if face_data['mp_landmarks'] is not None:
            # Draw MediaPipe landmarks
            lip_indices = self.UPPER_LIP_INDICES + self.LOWER_LIP_INDICES
            
            # Make dots larger for better visibility
            dot_size = 4 if face_data.get('is_wide_shot', False) else 2
            
            for idx in lip_indices:
                landmark = face_data['mp_landmarks'].landmark[idx]
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), dot_size, color, -1)
        
        # For wide shots or when MediaPipe fails, draw only mouth region rectangle
        elif 'mouth_region' in face_data:
            mouth = face_data['mouth_region']
            # Draw mouth region rectangle with transparent fill (only borders)
            cv2.rectangle(frame, 
                         (mouth['x1'], mouth['y1']), 
                         (mouth['x2'], mouth['y2']), 
                         color, 2)  # Increased thickness for better visibility
        
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
        
        
        return is_wide 