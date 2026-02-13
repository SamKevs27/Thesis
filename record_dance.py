import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- MATH SECTION ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle 

cap = cv2.VideoCapture(0)

# RECORDING VARIABLES
recording = False
dance_data = [] 
frame_count = 0

print("Press 'r' to START/STOP recording.")
print("Press 'q' to QUIT.")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        h, w, _ = image.shape

        try:
            landmarks = results.pose_landmarks.landmark
            
            # --- GET COORDINATES ---
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow    = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist    = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow    = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist    = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            l_hip      = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee     = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_ankle    = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            r_hip      = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee     = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            r_ankle    = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # --- CALCULATE ANGLES (Now rounded to integers) ---
            # We use int() here to strip the decimals
            angles = [
                int(calculate_angle(l_shoulder, l_elbow, l_wrist)),  # L_Elbow
                int(calculate_angle(r_shoulder, r_elbow, r_wrist)),  # R_Elbow
                int(calculate_angle(l_hip, l_shoulder, l_elbow)),    # L_Shoulder
                int(calculate_angle(r_hip, r_shoulder, r_elbow)),    # R_Shoulder
                int(calculate_angle(l_shoulder, l_hip, l_knee)),     # L_Hip
                int(calculate_angle(r_shoulder, r_hip, r_knee)),     # R_Hip
                int(calculate_angle(l_hip, l_knee, l_ankle)),        # L_Knee
                int(calculate_angle(r_hip, r_knee, r_ankle))         # R_Knee
            ]

            # --- RECORDING LOGIC ---
            if recording:
                dance_data.append(angles)
                frame_count += 1
                cv2.putText(image, f"RECORDING: {frame_count} frames", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # UI Instructions
        if not recording:
            cv2.putText(image, "Press 'r' to Record", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Thesis Recorder', image)

        # Keyboard Controls
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            if recording:
                # STOP RECORDING
                recording = False
                print(f"Recording stopped. Saved {len(dance_data)} frames.")
                
                # Save to CSV
                with open('dance_data.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Header row
                    writer.writerow(['LElbow','RElbow','LShoulder','RShoulder','LHip','RHip','LKnee','RKnee'])
                    # Data rows
                    writer.writerows(dance_data)
                print("File 'dance_data.csv' saved successfully!")
                dance_data = [] 
                frame_count = 0
            else:
                # START RECORDING
                recording = True
                print("Recording started...")
                dance_data = []

cap.release()
cv2.destroyAllWindows()