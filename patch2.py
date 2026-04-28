import re

with open("app.py", "r") as f:
    content = f.read()

# Add import
import_stmt = "from flask import Flask, render_template, request, jsonify, session"
new_import = "from flask import Flask, render_template, request, jsonify, session\nfrom pose_classifier import PoseClassifier"
content = content.replace(import_stmt, new_import)


old_extract_start = """    c = np.array(c)
    
    # Create vectors meeting at point b"""

new_extract_start = """    c = np.array(c)
    
    # Create vectors meeting at point b"""
# not touching calculate_angle


old_loop = """          timestamps.append(raw_frame_index / float(fps))

         except Exception:
          # ignore per-frame errors
          pass"""

new_loop = """          timestamps.append(raw_frame_index / float(fps))
          
          # POSE CLASSIFICATION
          LH = PL.LEFT_HIP.value; RH = PL.RIGHT_HIP.value
          LHEEL = PL.LEFT_HEEL.value; RHEEL = PL.RIGHT_HEEL.value
          hip_y = (lm[LH].y + lm[RH].y) / 2.0
          heel_y = (lm[LHEEL].y + lm[RHEEL].y) / 2.0
          
          current_angles = {
              'left_knee': row[8],
              'right_knee': row[9],
              'left_hip': row[6],
              'right_hip': row[7],
              'left_shoulder': row[2],
              'right_shoulder': row[3]
          }
          current_state = classifier.detect_pose(current_angles, heel_y=heel_y, hip_y=hip_y)
          pose_labels.append(current_state)

         except Exception:
          # ignore per-frame errors
          pass"""
content = content.replace(old_loop, new_loop)

old_ret = """    if not dance_data:
     return None, None, None, "No pose detected in video"

    # Smooth the extracted angle data using the Savitzky-Golay offline filter
    smoothed_data = smooth_pose_data(np.array(dance_data))

    return smoothed_data, np.array(timestamps), landmarks_data, None"""

new_ret = """    if not dance_data:
     return None, None, None, None, "No pose detected in video"

    # Smooth the extracted angle data using the Savitzky-Golay offline filter
    smoothed_data = smooth_pose_data(np.array(dance_data))

    return smoothed_data, np.array(timestamps), landmarks_data, pose_labels, None"""
content = content.replace(old_ret, new_ret)

old_init_lists = """    dance_data = []
    timestamps = []
    landmarks_data = [] # New: store (x, y, visibility) for overlay
    raw_frame_index = 0"""

new_init_lists = """    dance_data = []
    timestamps = []
    landmarks_data = [] # New: store (x, y, visibility) for overlay
    pose_labels = []    # Store detected pose states
    raw_frame_index = 0
    classifier = PoseClassifier()"""
content = content.replace(old_init_lists, new_init_lists)

old_call1 = """        # Extract angles and timestamps from saved teacher video
        dance_data, timestamps, landmarks_data, error = extract_angles_from_video(filepath)

        if error:
            os.remove(filepath)
            return jsonify({'error': error}), 400"""

new_call1 = """        # Extract angles and timestamps from saved teacher video
        dance_data, timestamps, landmarks_data, pose_labels, error = extract_angles_from_video(filepath)

        if error:
            os.remove(filepath)
            return jsonify({'error': error}), 400
        
        # Save pose labels
        import json
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'teacher_poses.json'), 'w') as f:
            json.dump(pose_labels, f)"""
content = content.replace(old_call1, new_call1)

old_call2 = """        # Extract angles from video (also get timestamps)
        student_data, student_ts, landmarks_data, error = extract_angles_from_video(filepath)

        if error:"""

new_call2 = """        # Extract angles from video (also get timestamps)
        student_data, student_ts, landmarks_data, pose_labels, error = extract_angles_from_video(filepath)

        if error:"""
content = content.replace(old_call2, new_call2)

old_call3 = """        # Also save student landmarks
        import json
        with open(os.path.join(app.config['UPLOAD_FOLDER'], f"student_{student_name}_landmarks.json"), 'w') as f:
            json.dump(landmarks_data, f)"""

new_call3 = """        # Also save student landmarks and poses
        import json
        with open(os.path.join(app.config['UPLOAD_FOLDER'], f"student_{student_name}_landmarks.json"), 'w') as f:
            json.dump(landmarks_data, f)
        with open(os.path.join(app.config['UPLOAD_FOLDER'], f"student_{student_name}_poses.json"), 'w') as f:
            json.dump(pose_labels, f)"""
content = content.replace(old_call3, new_call3)


with open("app.py", "w") as f:
    f.write(content)

