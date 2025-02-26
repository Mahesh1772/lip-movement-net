import cv2
from lip_detector import LipDetector

def main():
    # Initialize detector
    detector = LipDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open webcam!")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        frame, is_speaking = detector.detect_speaking(frame)
        
        # Display the frame
        cv2.imshow('Lip Detection', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 