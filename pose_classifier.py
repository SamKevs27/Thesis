class PoseClassifier:
    """
    A heuristic rule-based pose classification engine.
    Uses 3D joint angles and Y-coordinates to determine basic poses.
    Thresholds can be easily tuned for specific dance styles like Hip-Hop.
    """
    
    def __init__(self):
        # Angle thresholds (in degrees, 0-180 format)
        
        # SQUAT THRESHOLDS
        # A deep squat usually has knees bent past 100 degrees,
        # but a hip-hop half-squat or groove bounce might be around 120-130.
        self.SQUAT_KNEE_MAX = 120.0 
        
        # STANDING THRESHOLDS
        # Straight legs are typically 160-180 degrees.
        self.STAND_KNEE_MIN = 160.0
        self.STAND_HIP_MIN = 150.0  # Hips relatively straight, not hinged forward
        
        # ARMS RAISED THRESHOLDS
        # Shoulders reaching upwards. Depending on how the angle is calculated
        # (e.g., hip-shoulder-elbow), raising arms usually exceeds 130-140 degrees.
        self.ARMS_RAISED_SHOULDER_MIN = 140.0

    def detect_pose(self, angles, heel_y=None, hip_y=None):
        """
        Evaluate the current joint angles and Y-coordinates to classify the pose.
        
        :param angles: Dictionary of 3D angles, e.g., 
                       {'left_knee': 170, 'right_knee': 170, 
                        'left_hip': 160, 'right_hip': 160,
                        'left_shoulder': 45, 'right_shoulder': 45}
        :param heel_y: Average Y-coordinate of the heels (optional, for spatial checks)
        :param hip_y: Average Y-coordinate of the hips (optional)
        :return: String label of the detected pose
        """
        
        # Extract angles with safe fallbacks (default to straight/neutral if missing)
        l_knee = angles.get('left_knee', 180.0)
        r_knee = angles.get('right_knee', 180.0)
        l_hip = angles.get('left_hip', 180.0)
        r_hip = angles.get('right_hip', 180.0)
        l_shoulder = angles.get('left_shoulder', 0.0)
        r_shoulder = angles.get('right_shoulder', 0.0)

        # Average bilateral angles for symmetric pose checks
        avg_knee = (l_knee + r_knee) / 2.0
        avg_hip = (l_hip + r_hip) / 2.0
        max_shoulder = max(l_shoulder, r_shoulder) # If AT LEAST one arm is raised

        # 1. CHECK SQUATTING
        # If the average knee angle is sharply bent (less than the max threshold).
        # We can also cross-reference heel_y and hip_y if provided (in MediaPipe, lower Y value means higher on screen).
        is_squatting = avg_knee < self.SQUAT_KNEE_MAX
        
        # Optional spatial verification: distance between hips and heels decreases in a squat.
        # This is useful to distinguish a squat from simply lifting the knees while seated.
        if heel_y is not None and hip_y is not None:
            spatial_distance = heel_y - hip_y 
            # If the distance shrinks significantly, it reinforces the squat prediction.
            # (Requires calibration based on bounding box or normalized coordinates)

        if is_squatting:
            return "Squatting"

        # 2. CHECK ARMS RAISED
        # If knees are not squatted, check if the dancer is hitting a high-V or reaching up.
        is_arms_raised = max_shoulder > self.ARMS_RAISED_SHOULDER_MIN
        
        if is_arms_raised:
            return "Arms Raised"

        # 3. CHECK STANDING STRAIGHT
        # Legs are straight (high knee angle) and torso is upright (high hip angle).
        is_standing = avg_knee > self.STAND_KNEE_MIN and avg_hip > self.STAND_HIP_MIN
        
        if is_standing:
            return "Standing Straight"

        # 4. FALLBACK STATE
        # If they are slightly bent but not into a full squat, and arms are down.
        # Perfect for mid-groove hip-hop states.
        return "Transition / Neutral"
