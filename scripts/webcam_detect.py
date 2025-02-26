import cv2
import dlib
import numpy as np
from collections import deque
import time

def detect_lips_live():
    # Initialize face detector and facial landmarks predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open webcam!")
        return
    
    # Parameters for speaking detection
    SILENCE_THRESHOLD = 13.0  # Minimum height for speaking
    SILENCE_DURATION = 1.5    # Seconds of closed lips to consider silent
    
    # Buffer for temporal analysis
    last_speaking_time = time.time()
    is_currently_speaking = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect faces
        faces = detector(frame, 0)
        current_time = time.time()
        
        for face in faces:
            # Get facial landmarks
            landmarks = predictor(frame, face)
            
            # Extract lip points
            lip_points = []
            for n in range(48, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                lip_points.append((x, y))
            
            # Draw lip contour (make it thinner)
            lip_points = np.array(lip_points)
            cv2.polylines(frame, [lip_points], True, (0, 255, 0), 1)
            
            # Calculate lip height
            top_lip = np.array([landmarks.part(61).y, landmarks.part(62).y, landmarks.part(63).y])
            bottom_lip = np.array([landmarks.part(67).y, landmarks.part(66).y, landmarks.part(65).y])
            lip_height = np.mean(np.abs(top_lip - bottom_lip))
            
            # Update speaking status
            if lip_height > SILENCE_THRESHOLD:
                last_speaking_time = current_time
                is_currently_speaking = True
            elif current_time - last_speaking_time > SILENCE_DURATION:
                is_currently_speaking = False
            
            # Display info in smaller text
            cv2.putText(frame, f"Height: {lip_height:.1f}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display speaking status with color
            status_color = (0, 255, 0) if is_currently_speaking else (0, 0, 255)
            status_text = "Speaking" if is_currently_speaking else "Silent"
            cv2.putText(frame, status_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Display the frame
        cv2.imshow('Lip Detection', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_lips_live() 