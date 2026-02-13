import cv2
import mediapipe as mp
import numpy as np
import csv

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# CONFIGURATION
VIDEO_PATH = 'teacher_dance.mp4'  # VIDEO FILENAME HERE
OUTPUT_FILE = 'reference_data.csv'

# MATH HELPER
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle 

# Setup Video Capture
cap = cv2.VideoCapture(VIDEO_PATH)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

dance_data = []
frame_count = 0

print(f"Processing {VIDEO_PATH}...")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video
        
        # Resize large 4K videos to speed up processing
        frame = cv2.resize(frame, (640, 480))

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        
        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            
            # GET COORDINATES
            l_sh = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            l_el = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            l_wr = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,    landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
            r_sh = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            r_el = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            r_wr = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            l_hi = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,      landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            l_kn = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,     landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            l_an = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            r_hi = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,      landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            r_kn = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            r_an = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

            # CALCULATE 8 ANGLES
            row = [
                calculate_angle(l_sh, l_el, l_wr), # L_Elbow
                calculate_angle(r_sh, r_el, r_wr), # R_Elbow
                calculate_angle(l_hi, l_sh, l_el), # L_Shoulder
                calculate_angle(r_hi, r_sh, r_el), # R_Shoulder
                calculate_angle(l_sh, l_hi, l_kn), # L_Hip
                calculate_angle(r_sh, r_hi, r_kn), # R_Hip
                calculate_angle(l_hi, l_kn, l_an), # L_Knee
                calculate_angle(r_hi, r_kn, r_an)  # R_Knee
            ]
            dance_data.append(row)
            frame_count += 1
            
            # Draw skeleton
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        except:
            pass # Skip frames where body isn't detected

        # Show the video while processing
        cv2.imshow('Processing Reference Video...', image)
        
        # Press 'q' to stop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# SAVE TO CSV
if dance_data:
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['LElbow','RElbow','LShoulder','RShoulder','LHip','RHip','LKnee','RKnee'])
        writer.writerows(dance_data)
    print(f"SUCCESS: Processed {frame_count} frames.")
    print(f"Data saved to: {OUTPUT_FILE}")
else:
    print("Failed: No data extracted.")