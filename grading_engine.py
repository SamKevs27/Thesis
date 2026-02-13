import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- MATH SECTION ---
def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Vertex (The joint)
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle 

cap = cv2.VideoCapture(0)

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
            
            # --- 1. GET COORDINATES (Standard) ---
            # Arms
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow    = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist    = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow    = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist    = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Legs & Hips
            l_hip      = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee     = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_ankle    = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            r_hip      = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee     = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            r_ankle    = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # --- 2. CALCULATE 8 KEY ANGLES ---
            
            # ELBOWS (Arm straightness)
            angle_l_elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)
            angle_r_elbow = calculate_angle(r_shoulder, r_elbow, r_wrist)
            
            # SHOULDERS (Arm Lift/Armpit angle) -- NEW!
            angle_l_shoulder = calculate_angle(l_hip, l_shoulder, l_elbow)
            angle_r_shoulder = calculate_angle(r_hip, r_shoulder, r_elbow)
            
            # KNEES (Leg straightness)
            angle_l_knee = calculate_angle(l_hip, l_knee, l_ankle)
            angle_r_knee = calculate_angle(r_hip, r_knee, r_ankle)
            
            # HIPS (Torso bend/Leg lift) -- NEW!
            angle_l_hip = calculate_angle(l_shoulder, l_hip, l_knee)
            angle_r_hip = calculate_angle(r_shoulder, r_hip, r_knee)

            # --- 3. VISUALIZATION ---
            
            # Helper to draw text at joint
            def draw_angle(img, coords, angle, color=(255,255,255)):
                cv2.putText(img, str(int(angle)), 
                           tuple(np.multiply(coords, [w, h]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # Draw Elbow Angles
            draw_angle(image, l_elbow, angle_l_elbow)
            draw_angle(image, r_elbow, angle_r_elbow)
            
            # Draw Shoulder Angles (Yellow for contrast)
            draw_angle(image, l_shoulder, angle_l_shoulder, (0, 255, 255))
            draw_angle(image, r_shoulder, angle_r_shoulder, (0, 255, 255))
            
            # Draw Knee Angles
            draw_angle(image, l_knee, angle_l_knee)
            draw_angle(image, r_knee, angle_r_knee)
            
            # Draw Hip Angles (Yellow)
            draw_angle(image, l_hip, angle_l_hip, (0, 255, 255))
            draw_angle(image, r_hip, angle_r_hip, (0, 255, 255))

            # --- 4. REPORT CARD (Updated for 8 Joints) ---
            cv2.rectangle(image, (0,0), (250, 280), (245, 117, 16), -1)
            cv2.putText(image, "JOINT REPORT", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            
            y_pos = 50
            for name, val in [("L-Elbow", angle_l_elbow), ("R-Elbow", angle_r_elbow),
                              ("L-Shoulder", angle_l_shoulder), ("R-Shoulder", angle_r_shoulder),
                              ("L-Hip", angle_l_hip), ("R-Hip", angle_r_hip),
                              ("L-Knee", angle_l_knee), ("R-Knee", angle_r_knee)]:
                
                cv2.putText(image, f"{name}: {int(val)}", (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                y_pos += 30

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Thesis Full-Body Engine', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()