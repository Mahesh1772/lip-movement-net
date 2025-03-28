import cv2
import numpy as np
import threading
import queue
from datetime import datetime, timedelta
import time
from lip_detector import MediaPipeLipDetector

class MultiCamLipDetector:
    def __init__(self, phone_ip):
        self.phone_url = f"http://{phone_ip}:8080/video"
        print(f"\nTrying to connect to phone camera at: {self.phone_url}\n")
        
        # Initialize frame queues
        self.webcam_queue = queue.Queue(maxsize=2)
        self.phone_queue = queue.Queue(maxsize=2)
        
        # Initialize lip detectors (one for each feed)
        self.webcam_detector = MediaPipeLipDetector()
        self.phone_detector = MediaPipeLipDetector()
        
        self.running = True
        print("Initialization complete. Starting capture...")

    def process_feeds(self):
        try:
            print("Starting display loop...")
            cv2.namedWindow('Camera Feeds', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Speaking Output', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Camera Feeds', 1280, 360)
            cv2.resizeWindow('Speaking Output', 640, 360)
            
            while self.running:
                try:
                    # Get frames from queues
                    webcam_frame = self.webcam_queue.get(timeout=0.1)
                    phone_frame = self.phone_queue.get(timeout=0.1)
                    
                    # Process frames with lip detection
                    webcam_processed, webcam_speaking = self.webcam_detector.detect_speaking(webcam_frame)
                    phone_processed, phone_speaking = self.phone_detector.detect_speaking(phone_frame)
                    
                    # Create output frame for speaking persons
                    output_frames = []
                    
                    # Add webcam frame if someone is speaking
                    if any(webcam_speaking.values()):
                        webcam_resized = cv2.resize(webcam_processed, (640, 360))
                        cv2.putText(webcam_resized, "Laptop Camera", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        output_frames.append(webcam_resized)
                    
                    # Add phone frame if someone is speaking
                    if any(phone_speaking.values()):
                        phone_resized = cv2.resize(phone_processed, (640, 360))
                        cv2.putText(phone_resized, "Phone Camera", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        output_frames.append(phone_resized)
                    
                    # Display speaking output
                    if output_frames:
                        if len(output_frames) == 1:
                            output_view = output_frames[0]
                        else:
                            output_view = np.hstack(output_frames)
                        cv2.imshow('Speaking Output', output_view)
                    else:
                        # Show blank frame when no one is speaking
                        blank_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(blank_frame, "No one speaking", (200, 180), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.imshow('Speaking Output', blank_frame)
                    
                    # Display main camera feeds
                    webcam_resized = cv2.resize(webcam_processed, (640, 360))
                    phone_resized = cv2.resize(phone_processed, (640, 360))
                    debug_view = np.hstack((webcam_resized, phone_resized))
                    
                    # Add labels to main view
                    cv2.putText(debug_view, "Laptop Webcam", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(debug_view, "Phone Camera", (650, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Show the combined view
                    cv2.imshow('Camera Feeds', debug_view)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Display error: {str(e)}")
                    continue
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.running = False
            cv2.destroyAllWindows()

    def capture_webcam(self):
        print("Opening laptop webcam...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("Error: Could not open laptop webcam!")
            return
                
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"Laptop webcam opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                if self.webcam_queue.full():
                    try:
                        self.webcam_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.webcam_queue.put(frame)
            else:
                print("Failed to grab webcam frame")
                time.sleep(0.1)
        cap.release()

    def capture_phone(self):
        print(f"Opening phone camera stream from: {self.phone_url}")
        cap = cv2.VideoCapture(self.phone_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print("Error: Could not open phone camera stream!")
            return
                
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"Phone camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                if self.phone_queue.full():
                    try:
                        self.phone_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.phone_queue.put(frame)
            else:
                print("Failed to grab phone frame")
                time.sleep(0.1)
        cap.release()

    def run(self):
        # Start capture threads
        webcam_thread = threading.Thread(target=self.capture_webcam)
        phone_thread = threading.Thread(target=self.capture_phone)
        
        webcam_thread.start()
        phone_thread.start()
        
        # Run processing in main thread
        self.process_feeds()
        
        # Wait for capture threads to finish
        webcam_thread.join()
        phone_thread.join()

if __name__ == "__main__":
    detector = MultiCamLipDetector("192.168.2.155")
    detector.run() 