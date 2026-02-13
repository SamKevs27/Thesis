import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Flip frame for mirror effect (optional, feels more natural)
    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions for drawing lines
    img_h, img_w, _ = image.shape

    # 1. Draw Standard Landmarks
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

    # --- THE FIX: MANUAL WRIST CONNECTIONS ---
    if results.pose_landmarks:
        # Get Pose Wrists (Index 15=Left, 16=Right)
        # Note: "Left" pose is typically your right hand in mirrored view, 
        # but MediaPipe keeps the naming consistent with the person's body.
        
        # CONNECT LEFT HAND (If visible)
        if results.left_hand_landmarks:
            # 1. Get Pose Wrist Coordinates
            pose_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
            px, py = int(pose_wrist.x * img_w), int(pose_wrist.y * img_h)
            
            # 2. Get Hand Wrist Coordinates (Index 0 is always the wrist)
            hand_wrist = results.left_hand_landmarks.landmark[0]
            hx, hy = int(hand_wrist.x * img_w), int(hand_wrist.y * img_h)
            
            # 3. Draw Line
            cv2.line(image, (px, py), (hx, hy), (255, 255, 255), 3)

        # CONNECT RIGHT HAND (If visible)
        if results.right_hand_landmarks:
            pose_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
            px, py = int(pose_wrist.x * img_w), int(pose_wrist.y * img_h)
            
            hand_wrist = results.right_hand_landmarks.landmark[0]
            hx, hy = int(hand_wrist.x * img_w), int(hand_wrist.y * img_h)
            
            cv2.line(image, (px, py), (hx, hy), (255, 255, 255), 3)

    cv2.imshow('Thesis Holistic Connected', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()