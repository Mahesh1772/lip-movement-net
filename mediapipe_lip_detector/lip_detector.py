import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict, deque
import time

class MediaPipeLipDetector:
    def __init__(self):
        # Tell TensorFlow to use GPU if available (helps MediaPipe)
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"TensorFlow using GPU: {gpus}")
            else:
                print("No GPU found, using CPU")
        except Exception as e:
            print(f"Error configuring TensorFlow GPU: {e}")
        
        # Initialize MediaPipe for face detection and lip tracking
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=8,
            refine_landmarks=True,
            min_detection_confidence=0.6,  # Increased from 0.3 to 0.6
            min_tracking_confidence=0.5    # Increased from 0.3 to 0.5
        )
        
        # MediaPipe lip indices
        self.UPPER_LIP_INDICES = [13, 14, 312]  # Upper lip indices in MediaPipe
        self.LOWER_LIP_INDICES = [17, 16, 15]   # Lower lip indices in MediaPipe
        
        # Parameters for speaking detection
        self.SILENCE_THRESHOLD = 0.04           # Threshold for open mouth
        self.MOVEMENT_THRESHOLD = 0.01          # Threshold for general lip movement
        self.LOOKING_DOWN_THRESHOLD = 0.007     # Lower threshold for when looking down
        self.SILENCE_DURATION = 2.5             # Silence duration
        self.SPEAKING_FRAMES_THRESHOLD = 6      # Required consistent movement frames
        self.DECAY_RATE = 0.3                   # Slower decay
        self.MIN_SPEAKING_DURATION = 1.0        # Minimum speaking duration
        self.HEAD_MOVEMENT_THRESHOLD = 0.015    # Threshold for head movement detection
        self.HEAD_MOVEMENT_FACTOR = 10          # Factor to reduce lip movement during head motion
        self.ANOMALY_THRESHOLD = 2.5            # Threshold for detecting anomalous movements
        self.LOOKING_DOWN_ANGLE = 30            # Threshold angle in degrees for looking down
        
        # Face tracking system
        self.face_histories = {}
        
        # For consistent face IDs display
        self.next_display_id = 0
        self.id_mapping = {}
        self.active_display_ids = set()
        
        # Frame counter
        self.frame_count = 0
        
        # Debug flag
        self.debug = False
    
    def get_faces_and_lips(self, image):
        # Process with MediaPipe for face detection and lip landmarks
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_results = self.mp_face_mesh.process(rgb_image)
        
        current_faces = []
        
        if mp_results and mp_results.multi_face_landmarks:
            for i, face_landmarks in enumerate(mp_results.multi_face_landmarks):
                # Get face bounding box
                landmarks = face_landmarks.landmark
                h, w = image.shape[:2]
                
                # Extract face bounding box using landmarks
                x_coordinates = [landmark.x * w for landmark in landmarks]
                y_coordinates = [landmark.y * h for landmark in landmarks]
                x1, y1 = int(min(x_coordinates)), int(min(y_coordinates))
                x2, y2 = int(max(x_coordinates)), int(max(y_coordinates))
                
                # Add padding to the bounding box
                padding_x = int((x2 - x1) * 0.05)
                padding_y = int((y2 - y1) * 0.05)
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(w, x2 + padding_x)
                y2 = min(h, y2 + padding_y)
                
                # Calculate lip height using MediaPipe landmarks
                lip_height = self.calculate_lip_height_mediapipe(face_landmarks, image.shape)
                
                # Store detection
                current_faces.append({
                    'id': i,
                    'height': lip_height,
                    'bbox': [x1, y1, x2, y2, 1.0],  # [x1, y1, x2, y2, confidence]
                    'mp_landmarks': face_landmarks,
                    'face_size': (x2 - x1) * (y2 - y1),  # Add face size info
                })
        
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
        
        # If no faces detected, immediately mark all existing faces as not speaking
        if not faces_data:
            # Mark all existing face histories as not speaking
            for face_id in self.face_histories:
                self.face_histories[face_id]['is_speaking'] = False
                self.face_histories[face_id]['speaking_confidence'] = 0.0
                self.face_histories[face_id]['speaking_frames_count'] = 0
            
            # Return the frame unchanged with no speaking faces
            return frame, {}
        
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
                    'speaking_start_time': current_time,
                    'head_position': deque(maxlen=5),  # Add head position tracking
                    'last_variation': deque(maxlen=10),  # Add variation history
                    'face_size': face_data.get('face_size', 0.0),  # Store face size
                    'consecutive_moving_frames': 0,  # Track consecutive frames with movement
                    'consecutive_still_frames': 0,   # Track consecutive frames without movement
                    'speaking_confidence': 0.0,       # Add confidence score
                    'is_looking_down': False
                }
            else:
                # Update display ID in history
                self.face_histories[face_id]['display_id'] = display_id
                # Update face size
                self.face_histories[face_id]['face_size'] = face_data.get('face_size', 0.0)
            
            face_history = self.face_histories[face_id]
            
            # Get MediaPipe landmarks
            mp_landmarks = face_data.get('mp_landmarks')
            
            # Variables to track head orientation
            is_profile_view = False
            is_looking_down = False
            
            if mp_landmarks:
                # Track head position to filter out head movement
                nose_x = mp_landmarks.landmark[1].x
                nose_y = mp_landmarks.landmark[1].y
                face_history['head_position'].append((nose_x, nose_y))
                
                # Calculate head movement
                head_movement = 0
                if len(face_history['head_position']) >= 3:
                    positions = list(face_history['head_position'])
                    head_movement = np.mean([
                        np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                (positions[i][1] - positions[i-1][1])**2)
                        for i in range(1, len(positions))
                    ])
                
                # Check face orientation - detect profile views
                if len(mp_landmarks.landmark) > 0:
                    # Check horizontal face orientation using ear and nose landmarks
                    left_ear = mp_landmarks.landmark[234]  # Left ear landmark
                    right_ear = mp_landmarks.landmark[454]  # Right ear landmark
                    nose = mp_landmarks.landmark[1]  # Nose tip landmark
                    
                    # If one ear is much more visible than the other, it's likely a profile view
                    ear_diff = abs(left_ear.z - right_ear.z)
                    if ear_diff > 0.1:  # Significant depth difference between ears
                        is_profile_view = True
                    
                    # Detect looking down by comparing eye and chin position
                    # When looking down, the eyes will be lower relative to the chin
                    left_eye = mp_landmarks.landmark[33]  # Left eye
                    right_eye = mp_landmarks.landmark[263]  # Right eye
                    chin = mp_landmarks.landmark[152]  # Chin
                    
                    # Calculate the angle between the eye-nose line and horizontal
                    eye_y = (left_eye.y + right_eye.y) / 2.0
                    eye_x = (left_eye.x + right_eye.x) / 2.0
                    
                    # Vector from eyes to chin
                    eye_to_chin_x = chin.x - eye_x
                    eye_to_chin_y = chin.y - eye_y
                    
                    # Calculate angle in degrees (0 is horizontal, positive is looking down)
                    # We're mainly interested in the vertical angle
                    angle = np.degrees(np.arctan2(eye_to_chin_y, max(abs(eye_to_chin_x), 0.001)))
                    
                    # Alternative detection: check if nose is visibly below eyes
                    nose_below_eyes = nose.y > eye_y + 0.02
                    
                    # Consider looking down if angle is large enough
                    is_looking_down = angle > self.LOOKING_DOWN_ANGLE or nose_below_eyes
                    
                    # Store the looking down status in face history
                    face_history['is_looking_down'] = is_looking_down
            else:
                head_movement = 0
                is_profile_view = False
                is_looking_down = False
            
            # Use lip height for speaking detection
            lip_height = face_data.get('height', 0.0)
            face_history['last_heights'].append(lip_height)
            
            if len(face_history['last_heights']) >= 5:
                recent_heights = list(face_history['last_heights'])[-5:]
                variation = np.std(recent_heights)
                
                # Track recent variations
                face_history['last_variation'].append(variation)
                
                # Detect anomalies in variation (sudden spikes)
                is_anomaly = False
                if len(face_history['last_variation']) >= 5:
                    avg_variation = np.mean(list(face_history['last_variation'])[:-1])
                    if variation > avg_variation * self.ANOMALY_THRESHOLD:
                        is_anomaly = True
                
                # Adjust variation based on head movement
                adjusted_variation = variation
                if head_movement > self.HEAD_MOVEMENT_THRESHOLD:
                    adjusted_variation = max(0, variation - (head_movement - self.HEAD_MOVEMENT_THRESHOLD) * self.HEAD_MOVEMENT_FACTOR)
                
                # Ignore anomalies
                if is_anomaly:
                    adjusted_variation = 0
                
                # Select the appropriate threshold based on head orientation
                movement_threshold = self.MOVEMENT_THRESHOLD
                if is_looking_down:
                    # Use a lower threshold when looking down
                    movement_threshold = self.LOOKING_DOWN_THRESHOLD
                
                is_moving = adjusted_variation > movement_threshold
                is_open = lip_height > self.SILENCE_THRESHOLD
                
                # Update consecutive frame counters
                if is_moving and is_open:
                    face_history['consecutive_moving_frames'] += 1
                    face_history['consecutive_still_frames'] = 0
                else:
                    face_history['consecutive_moving_frames'] = 0
                    face_history['consecutive_still_frames'] += 1
                
                # Require at least 3 consecutive frames of movement to consider it real
                real_movement = face_history['consecutive_moving_frames'] >= 3
                
                # Debug info
                if self.debug and self.frame_count % 30 == 0:
                    print(f"\nFace {display_id} Debug Info:")
                    print(f"Lip Height: {lip_height:.3f} (Threshold: {self.SILENCE_THRESHOLD})")
                    print(f"Movement Variation: {variation:.3f} (Adjusted: {adjusted_variation:.3f})")
                    print(f"Head Movement: {head_movement:.5f} (Threshold: {self.HEAD_MOVEMENT_THRESHOLD})")
                    print(f"Is Looking Down: {is_looking_down} (Using threshold: {movement_threshold:.5f})")
                    print(f"Is Anomaly: {is_anomaly}")
                    print(f"Moving: {is_moving}, Open: {is_open}, Real Movement: {real_movement}")
                    print(f"Speaking Frames Count: {face_history['speaking_frames_count']}")
                    print(f"Speaking Confidence: {face_history['speaking_confidence']:.2f}")
                    print(f"Currently Speaking: {face_history['is_speaking']}")
                
                # Calculate profile adjustment
                profile_bonus = 0.02 if is_profile_view else 0.0
                looking_down_bonus = 0.01 if is_looking_down else 0.0
                
                if real_movement:
                    # Apply profile and looking down bonuses to increase speaking confidence
                    confidence_boost = 0.15 + profile_bonus + looking_down_bonus
                    face_history['speaking_frames_count'] += 1.5
                    face_history['speaking_confidence'] = min(1.0, face_history['speaking_confidence'] + confidence_boost)
                    
                    if face_history['speaking_frames_count'] >= self.SPEAKING_FRAMES_THRESHOLD:
                        if not face_history['is_speaking'] and face_history['speaking_confidence'] > 0.6:
                            # Just started speaking - record the time
                            face_history['speaking_start_time'] = current_time
                        face_history['last_speaking_time'] = current_time
                        face_history['is_speaking'] = True
                else:
                    # Slower decay when not moving
                    face_history['speaking_frames_count'] = max(0, face_history['speaking_frames_count'] - self.DECAY_RATE)
                    
                    # Gradually reduce confidence
                    confidence_decay = 0.05
                    if head_movement > self.HEAD_MOVEMENT_THRESHOLD * 2:
                        # Reduce confidence decay during significant head movement
                        confidence_decay = 0.02
                    
                    # Be more lenient with profile views
                    if is_profile_view:
                        confidence_decay = max(0.01, confidence_decay - 0.02)
                        
                    face_history['speaking_confidence'] = max(0.0, face_history['speaking_confidence'] - confidence_decay)
                    
                    # Only transition to silent if confidence is low enough and we've met other criteria
                    if (face_history['is_speaking'] and 
                        current_time - face_history['speaking_start_time'] > self.MIN_SPEAKING_DURATION and
                        current_time - face_history['last_speaking_time'] > self.SILENCE_DURATION and
                        face_history['consecutive_still_frames'] > 20 and
                        face_history['speaking_confidence'] < 0.3):
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
        
        # Color based on speaking status
        color = (0, 255, 0) if is_speaking else (0, 0, 255)
        
        # Draw face bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Get looking down status if available
        is_looking_down = False
        if face_data['id'] in self.face_histories:
            is_looking_down = self.face_histories[face_data['id']].get('is_looking_down', False)
        
        # Add speaking status text and looking down indicator
        status_text = "Speaking" if is_speaking else "Silent"
        if is_looking_down:
            status_text += " (↓)"  # Add down arrow to indicate looking down
        
        cv2.putText(frame, status_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw lip landmarks if MediaPipe landmarks are available
        mp_landmarks = face_data.get('mp_landmarks')
        if mp_landmarks is not None:
            # Draw MediaPipe landmarks
            lip_indices = self.UPPER_LIP_INDICES + self.LOWER_LIP_INDICES
            
            # Make dots larger for better visibility
            dot_size = 2
            
            for idx in lip_indices:
                landmark = mp_landmarks.landmark[idx]
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