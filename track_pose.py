import cv2
import mediapipe as mp
import numpy as np

# Try to access solutions. If this fails, we will see the error immediately.
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    print("Error: mp.solutions not found. Trying manual import...")
    # Fallback for some specific Python/MediaPipe version mismatches
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing

# Initialize Pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open Webcam
cap = cv2.VideoCapture(0)
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera frame empty")
        break

    # Recolor to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
  
    # Make detection
    results = pose.process(image)
  
    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow('Thesis Pose Feed', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()