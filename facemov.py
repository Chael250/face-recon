import cv2
import time

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to track previous position and time
prev_cx, prev_cy = None, None
prev_time = None

# Movement threshold (pixels)
THRESHOLD = 5

print("Face Movement Tracker Started!")
print("Press 'q' to quit")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Current time for speed calculation
    current_time = time.time()
    
    # Process detected faces
    if len(faces) > 0:
        # Use the first detected face
        x, y, w, h = faces[0]
        
        # Calculate center of the face
        cx = x + w // 2
        cy = y + h // 2
        
        # Draw green bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw center point
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        
        # Calculate movement if we have previous position
        if prev_cx is not None and prev_cy is not None and prev_time is not None:
            # Calculate displacement
            dx = cx - prev_cx
            dy = cy - prev_cy
            
            # Calculate time difference
            dt = current_time - prev_time
            
            # Determine direction
            direction = ""
            if abs(dx) > THRESHOLD or abs(dy) > THRESHOLD:
                if abs(dx) > abs(dy):
                    # Horizontal movement dominates
                    if dx > 0:
                        direction = "RIGHT"
                    else:
                        direction = "LEFT"
                else:
                    # Vertical movement dominates
                    if dy > 0:
                        direction = "DOWN"
                    else:
                        direction = "UP"
                
                # Calculate speed (pixels per second)
                distance = (dx**2 + dy**2)**0.5
                speed = distance / dt if dt > 0 else 0
                
                # Display movement info
                cv2.putText(frame, f"Direction: {direction}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Speed: {speed:.1f} px/s", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # No significant movement
                cv2.putText(frame, "Direction: STEADY", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Speed: 0.0 px/s", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # First frame with face detected
            cv2.putText(frame, "Direction: INITIALIZING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display face center coordinates
        cv2.putText(frame, f"Center: ({cx}, {cy})", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Update previous position and time
        prev_cx, prev_cy = cx, cy
        prev_time = current_time
    else:
        # No face detected
        cv2.putText(frame, "No face detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        prev_cx, prev_cy = None, None
        prev_time = None
    
    # Display FPS
    if prev_time is not None:
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show the frame
    cv2.imshow('Face Movement Tracker', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Face Movement Tracker Stopped!") 