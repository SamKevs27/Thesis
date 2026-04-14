"""
Flask web application for video-based dance grading system.
Allows uploading teacher and student videos for comparison.
"""

import os
import csv
import subprocess
import tempfile
import glob
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import correlate, resample
from scipy.io import wavfile
from datetime import datetime
import json
import shutil

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')
app.secret_key = 'dance_grading_secret_key_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Debugging
print(f"✓ Template folder: {os.path.abspath(app.template_folder)}")
print(f"✓ Static folder: {os.path.abspath(app.static_folder)}")

# MediaPipe setup
mp_pose = mp.solutions.pose

# ── Pose-highlight helpers ──────────────────────────────────────────────────
# Maps our angle joint names → the MediaPipe landmark IDs that define each joint.
_PL = mp_pose.PoseLandmark
JOINT_TO_LANDMARK_IDS: dict = {
    'L_Elbow':    [_PL.LEFT_SHOULDER.value,    _PL.LEFT_ELBOW.value,    _PL.LEFT_WRIST.value],
    'R_Elbow':    [_PL.RIGHT_SHOULDER.value,   _PL.RIGHT_ELBOW.value,   _PL.RIGHT_WRIST.value],
    'L_Shoulder': [_PL.LEFT_HIP.value,         _PL.LEFT_SHOULDER.value, _PL.LEFT_ELBOW.value],
    'R_Shoulder': [_PL.RIGHT_HIP.value,        _PL.RIGHT_SHOULDER.value,_PL.RIGHT_ELBOW.value],
    'L_Wrist':    [_PL.LEFT_ELBOW.value,       _PL.LEFT_WRIST.value,    _PL.LEFT_INDEX.value],
    'R_Wrist':    [_PL.RIGHT_ELBOW.value,      _PL.RIGHT_WRIST.value,   _PL.RIGHT_INDEX.value],
    'L_Hip':      [_PL.LEFT_SHOULDER.value,    _PL.LEFT_HIP.value,      _PL.LEFT_KNEE.value],
    'R_Hip':      [_PL.RIGHT_SHOULDER.value,   _PL.RIGHT_HIP.value,     _PL.RIGHT_KNEE.value],
    'L_Knee':     [_PL.LEFT_HIP.value,         _PL.LEFT_KNEE.value,     _PL.LEFT_ANKLE.value],
    'R_Knee':     [_PL.RIGHT_HIP.value,        _PL.RIGHT_KNEE.value,    _PL.RIGHT_ANKLE.value],
    'L_Ankle':    [_PL.LEFT_KNEE.value,        _PL.LEFT_ANKLE.value,    _PL.LEFT_FOOT_INDEX.value],
    'R_Ankle':    [_PL.RIGHT_KNEE.value,       _PL.RIGHT_ANKLE.value,   _PL.RIGHT_FOOT_INDEX.value],
    'Spine':      [_PL.LEFT_SHOULDER.value,    _PL.RIGHT_SHOULDER.value,
                   _PL.LEFT_HIP.value,         _PL.RIGHT_HIP.value,     _PL.NOSE.value],
}


def draw_pose_on_frame(frame, highlight_landmark_ids=None):
    """Draw MediaPipe pose skeleton on *frame* (BGR) and return an annotated copy.

    highlight_landmark_ids : list/set of MediaPipe landmark int IDs that
        correspond to the *problem* joint.  Those connections and dots are
        drawn in red/orange; all others are drawn in green.
    Returns the original frame unchanged on detection failure.
    """
    if frame is None:
        return frame

    highlight_set = set(highlight_landmark_ids or [])
    annotated = frame.copy()
    h, w = frame.shape[:2]

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4,
                          static_image_mode=True) as _pm:
            results = _pm.process(frame_rgb)

        if not (results and results.pose_landmarks):
            return annotated

        lm = results.pose_landmarks.landmark

        # Draw bone connections
        for conn in mp.solutions.pose.POSE_CONNECTIONS:
            s_id = int(conn[0])
            e_id = int(conn[1])
            lm_s = lm[s_id]
            lm_e = lm[e_id]
            if lm_s.visibility < 0.3 or lm_e.visibility < 0.3:
                continue
            pt1 = (int(lm_s.x * w), int(lm_s.y * h))
            pt2 = (int(lm_e.x * w), int(lm_e.y * h))
            is_hot    = s_id in highlight_set or e_id in highlight_set
            color     = (0, 60, 255) if is_hot else (50, 200, 50)
            thickness = 4           if is_hot else 2
            cv2.line(annotated, pt1, pt2, color, thickness)

        # Draw joint dots
        for idx, point in enumerate(lm):
            if point.visibility < 0.3:
                continue
            px, py = int(point.x * w), int(point.y * h)
            if idx in highlight_set:
                cv2.circle(annotated, (px, py), 10, (255, 255, 255), -1)
                cv2.circle(annotated, (px, py),  8, (0, 60, 255),    -1)
            else:
                cv2.circle(annotated, (px, py),  5, (255, 255, 255), -1)
                cv2.circle(annotated, (px, py),  4, (50, 200, 50),   -1)
    except Exception:
        pass

    return annotated
# ────────────────────────────────────────────────────────────────────────────


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import scipy.signal

def smooth_pose_data(pose_data, window_length=9, polyorder=3):
    """
    Smooths an entire sequence of pose angles using a Savitzky-Golay filter.
    """
    if len(pose_data) < window_length:
        window_length = len(pose_data) if len(pose_data) % 2 != 0 else len(pose_data) - 1
        
    if window_length <= polyorder:
        return pose_data # Sequence too short to smooth safely

    # Apply filter along the time axis
    smoothed_data = scipy.signal.savgol_filter(pose_data, window_length=window_length, polyorder=polyorder, axis=0)
    
    return smoothed_data

def calculate_angle(a, b, c):
    """
    Calculate the 3D interior angle between three spatial points.
    :param a: First point (e.g. Hip) [x, y, z]
    :param b: Middle point (e.g. Knee) [x, y, z]
    :param c: Last point (e.g. Ankle) [x, y, z]
    :return: Angle in degrees (0 to 180)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Create vectors meeting at point b
    ba = a - b
    bc = c - b
    
    # Calculate the dot product
    dot_prod = np.dot(ba, bc)
    
    # Calculate the magnitudes (norms) of the vectors
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    # Avoid division by zero
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
        
    # Calculate the cosine of the angle
    cosine_angle = dot_prod / (norm_ba * norm_bc)
    
    # Clip to handle floating point inaccuracies before arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Calculate the angle in degrees
    angle_deg = np.degrees(np.arccos(cosine_angle))
    
    return int(angle_deg)

def extract_angles_from_video(video_path):
    """Extract dance angles from video and return per-frame timestamps.

    Returns (dance_data_array, timestamps_array, None) on success or
    (None, None, error_message) on failure.

    Visibility gating: each angle is only calculated when all three of its
    constituent MediaPipe landmarks have visibility >= VIS_THRESHOLD.  When any
    landmark is occluded / uncertain the angle falls back to the most recent
    reliable value so the pose array stays dense (no NaN gaps).
    """
    VIS_THRESHOLD = 0.60   # 60% confidence required to trust a landmark

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
     return None, None, None, "Could not open video file"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    dance_data = []
    timestamps = []
    landmarks_data = [] # New: store (x, y, visibility) for overlay
    raw_frame_index = 0
    prev_row = None   # last fully-computed row — used for fallback angles

    def vis_ok(lm, *ids):
        """Return True if every landmark id in ids has visibility >= VIS_THRESHOLD."""
        return all(lm[i].visibility >= VIS_THRESHOLD for i in ids)

    def safe_angle(a, b, c, lm, *ids):
        """Return calculate_angle(a,b,c) if all landmark ids are visible, else None."""
        if vis_ok(lm, *ids):
            return calculate_angle(a, b, c)
        return None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
     while cap.isOpened():
         ret, frame = cap.read()
         if not ret:
          break

         raw_frame_index += 1

         # Resize for faster processing
         frame = cv2.resize(frame, (640, 480))

         # Convert to RGB
         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         image.flags.writeable = False
         results = pose.process(image)
         image.flags.writeable = True

         try:
          if results.pose_landmarks is None:
              continue

          lm = results.pose_landmarks.landmark
          PL = mp_pose.PoseLandmark

          # Convenience: grab [x,y,z] lists (only used when visibility check passes)
          def xy(lm_id):
              return [lm[lm_id.value].x, lm[lm_id.value].y, lm[lm_id.value].z]

          l_sh = xy(PL.LEFT_SHOULDER);   r_sh = xy(PL.RIGHT_SHOULDER)
          l_el = xy(PL.LEFT_ELBOW);      r_el = xy(PL.RIGHT_ELBOW)
          l_wr = xy(PL.LEFT_WRIST);      r_wr = xy(PL.RIGHT_WRIST)
          l_hi = xy(PL.LEFT_HIP);        r_hi = xy(PL.RIGHT_HIP)
          l_kn = xy(PL.LEFT_KNEE);       r_kn = xy(PL.RIGHT_KNEE)
          l_an = xy(PL.LEFT_ANKLE);      r_an = xy(PL.RIGHT_ANKLE)
          l_idx = xy(PL.LEFT_INDEX);     r_idx = xy(PL.RIGHT_INDEX)
          l_ft  = xy(PL.LEFT_FOOT_INDEX); r_ft = xy(PL.RIGHT_FOOT_INDEX)
          nose  = xy(PL.NOSE)
          mid_sh = [(l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2, (l_sh[2]+r_sh[2])/2]
          mid_hi = [(l_hi[0]+r_hi[0])/2, (l_hi[1]+r_hi[1])/2, (l_hi[2]+r_hi[2])/2]

          # landmark id shortcuts for readability
          LS = PL.LEFT_SHOULDER.value;   RS = PL.RIGHT_SHOULDER.value
          LE = PL.LEFT_ELBOW.value;      RE = PL.RIGHT_ELBOW.value
          LW = PL.LEFT_WRIST.value;      RW = PL.RIGHT_WRIST.value
          LH = PL.LEFT_HIP.value;        RH = PL.RIGHT_HIP.value
          LK = PL.LEFT_KNEE.value;       RK = PL.RIGHT_KNEE.value
          LA = PL.LEFT_ANKLE.value;      RA = PL.RIGHT_ANKLE.value
          LI = PL.LEFT_INDEX.value;      RI = PL.RIGHT_INDEX.value
          LF = PL.LEFT_FOOT_INDEX.value; RF = PL.RIGHT_FOOT_INDEX.value
          NO = PL.NOSE.value

          # Arms (0-5): L_Elbow, R_Elbow, L_Shoulder, R_Shoulder, L_Wrist, R_Wrist
          # Legs (6-11): L_Hip, R_Hip, L_Knee, R_Knee, L_Ankle, R_Ankle
          # Torso (12): Spine
          raw_row = [
              safe_angle(l_sh, l_el, l_wr,  lm, LS, LE, LW),   # 0  L_Elbow
              safe_angle(r_sh, r_el, r_wr,  lm, RS, RE, RW),   # 1  R_Elbow
              safe_angle(l_hi, l_sh, l_el,  lm, LH, LS, LE),   # 2  L_Shoulder
              safe_angle(r_hi, r_sh, r_el,  lm, RH, RS, RE),   # 3  R_Shoulder
              safe_angle(l_el, l_wr, l_idx, lm, LE, LW, LI),   # 4  L_Wrist
              safe_angle(r_el, r_wr, r_idx, lm, RE, RW, RI),   # 5  R_Wrist
              safe_angle(l_sh, l_hi, l_kn,  lm, LS, LH, LK),  # 6  L_Hip
              safe_angle(r_sh, r_hi, r_kn,  lm, RS, RH, RK),  # 7  R_Hip
              safe_angle(l_hi, l_kn, l_an,  lm, LH, LK, LA),  # 8  L_Knee
              safe_angle(r_hi, r_kn, r_an,  lm, RH, RK, RA),  # 9  R_Knee
              safe_angle(l_kn, l_an, l_ft,  lm, LK, LA, LF),  # 10 L_Ankle
              safe_angle(r_kn, r_an, r_ft,  lm, RK, RA, RF),  # 11 R_Ankle
              safe_angle(nose, mid_sh, mid_hi, lm, NO, LS, RS, LH, RH),  # 12 Spine
          ]

          # Fill None slots with previous frame's angle (temporal fall-back)
          row = []
          for col_i, val in enumerate(raw_row):
              if val is not None:
                  row.append(val)
              elif prev_row is not None:
                  row.append(prev_row[col_i])   # re-use last good value
              else:
                  row.append(90.0)              # neutral default first frame

          prev_row = row
          dance_data.append(row)
          # Save landmark coordinates for overlay
          lm_list = [{'x': l.x, 'y': l.y, 'v': l.visibility} for l in lm]
          landmarks_data.append(lm_list)
          timestamps.append(raw_frame_index / float(fps))

         except Exception:
          # ignore per-frame errors
          pass

    cap.release()

    if not dance_data:
     return None, None, None, "No pose detected in video"

    # Smooth the extracted angle data using the Savitzky-Golay offline filter
    smoothed_data = smooth_pose_data(np.array(dance_data))

    return smoothed_data, np.array(timestamps), landmarks_data, None

def load_teacher_data(filepath='dance_data.csv'):
    """Load teacher reference data"""
    try:
        data = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                data.append([int(float(x)) for x in row])
        # also try to load teacher timestamps if available
        ts_path = os.path.join(UPLOAD_FOLDER, 'teacher_timestamps.npy')
        if os.path.exists(ts_path):
            ts = np.load(ts_path)
        else:
            ts = None
        return np.array(data), ts, None
    except FileNotFoundError:
        return None, None, "Teacher reference not found"
    except Exception as e:
        return None, None, f"Error reading reference: {str(e)}"

def find_audio_offset(video_a_path: str, video_b_path: str,
                       sr: int = 16000, max_offset_sec: float = 60.0) -> float:
    """Return the time offset (seconds) between two videos based on audio cross-correlation.

    Positive offset  → video_a starts *later* in the music; trim video_a's pose by that amount.
    Negative offset  → video_b starts later in the music; trim video_b's pose.
    Returns 0.0 if audio extraction fails (graceful fallback).

    Parameters
    ----------
    sr            Sample rate for extraction (16 kHz is sufficient for beat detection).
    max_offset_sec  Only analyse the first N seconds of each audio stream to cap compute time.
    """
    tmp_a = tmp_b = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fa:
            tmp_a = fa.name
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fb:
            tmp_b = fb.name

        def _extract_wav(src, dst):
            cmd = [
                'ffmpeg', '-y', '-i', src,
                '-vn',                          # no video
                '-acodec', 'pcm_s16le',         # 16-bit PCM
                '-ar', str(sr),                 # target sample rate
                '-ac', '1',                     # mono
                '-t', str(max_offset_sec * 2),  # cap duration
                dst
            ]
            res = subprocess.run(cmd, capture_output=True, timeout=60)
            return res.returncode == 0

        ok_a = _extract_wav(video_a_path, tmp_a)
        ok_b = _extract_wav(video_b_path, tmp_b)
        if not ok_a or not ok_b:
            print('[align] Audio extraction failed; skipping alignment.')
            return 0.0

        # Read WAVs
        _, sig_a = wavfile.read(tmp_a)
        _, sig_b = wavfile.read(tmp_b)

        # Normalise to float32 [-1, 1]
        sig_a = sig_a.astype(np.float32) / (np.iinfo(np.int16).max + 1)
        sig_b = sig_b.astype(np.float32) / (np.iinfo(np.int16).max + 1)

        # Limit to max_offset_sec * 2 samples so FFT stays fast
        cap = int(max_offset_sec * 2 * sr)
        sig_a = sig_a[:cap]
        sig_b = sig_b[:cap]

        # Normalise energy (whitening makes correlation peak sharper)
        def _norm(s):
            s = s - s.mean()
            rms = np.sqrt((s ** 2).mean())
            return s / rms if rms > 1e-9 else s

        sig_a = _norm(sig_a)
        sig_b = _norm(sig_b)

        # FFT cross-correlation
        # correlate(a, b) → peak at lag L means a[t] ≈ b[t − L]
        # → b is shifted-right by L samples relative to a
        # → positive lag: video_a content appears L samples later → video_a started later
        corr = correlate(sig_a, sig_b, mode='full', method='fft')
        lags = np.arange(-(len(sig_b) - 1), len(sig_a))

        # Only consider lags within ±max_offset_sec to reject spurious peaks
        max_lag_samples = int(max_offset_sec * sr)
        valid_mask = np.abs(lags) <= max_lag_samples
        best_lag = lags[valid_mask][np.argmax(np.abs(corr[valid_mask]))]

        offset_sec = float(best_lag) / sr
        print(f'[align] Audio lag = {best_lag} samples → offset = {offset_sec:.3f}s  '
              f'(+ve ⇒ teacher starts later in the music)')
        return offset_sec

    except Exception as e:
        print(f'[align] find_audio_offset error: {e}; using 0.0')
        return 0.0
    finally:
        for p in [tmp_a, tmp_b]:
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception: pass


def align_pose_data(teacher_data, teacher_ts, student_data, student_ts, offset_sec: float):
    """Trim pose arrays so both start at the same musical beat.

    Sign convention (matches find_audio_offset):
      offset_sec > 0 → teacher video started later in the music
                       → drop the first |offset_sec| of teacher frames
      offset_sec < 0 → student video started later in the music
                       → drop the first |offset_sec| of student frames

    Returns (teacher_data, teacher_ts, student_data, student_ts, t_trim_sec, s_trim_sec)
    where t_trim_sec / s_trim_sec are the seconds trimmed from each side
    (used to adjust FFmpeg composition start times).
    """
    t_trim_sec = 0.0
    s_trim_sec = 0.0

    if abs(offset_sec) < 0.05:   # < 50 ms — negligible, skip
        return teacher_data, teacher_ts, student_data, student_ts, 0.0, 0.0

    if offset_sec > 0:
        # Teacher arrived late to the music → trim teacher's early (silent/different) frames
        t_trim_sec = offset_sec
        if teacher_ts is not None and len(teacher_ts) > 0:
            fps_t = len(teacher_ts) / float(teacher_ts[-1]) if teacher_ts[-1] > 0 else 30.0
        else:
            fps_t = 30.0
        trim_frames = int(round(offset_sec * fps_t))
        trim_frames = max(0, min(trim_frames, len(teacher_data) - 1))
        teacher_data = teacher_data[trim_frames:]
        if teacher_ts is not None:
            # FIX: preserve real-world timestamps — do NOT subtract base
            # DTW only cares about frame order; OpenCV needs the original time to seek correctly
            teacher_ts = teacher_ts[trim_frames:]
    else:
        # Student arrived late to the music → trim student's early frames
        s_trim_sec = abs(offset_sec)
        if student_ts is not None and len(student_ts) > 0:
            fps_s = len(student_ts) / float(student_ts[-1]) if student_ts[-1] > 0 else 30.0
        else:
            fps_s = 30.0
        trim_frames = int(round(abs(offset_sec) * fps_s))
        trim_frames = max(0, min(trim_frames, len(student_data) - 1))
        student_data = student_data[trim_frames:]
        if student_ts is not None:
            # FIX: preserve real-world timestamps — do NOT subtract base
            student_ts = student_ts[trim_frames:]

    return teacher_data, teacher_ts, student_data, student_ts, t_trim_sec, s_trim_sec


class BiomechanicalScorer:
    def __init__(self):
        # 13 Joints: [L_Elb, R_Elb, L_Sho, R_Sho, L_Wri, R_Wri, L_Hip, R_Hip, L_Kne, R_Kne, L_Ank, R_Ank, Spine]
        
        # 1. Biomechanical Weights (Core/Hips/Knees > Shoulders > Extemities/Wrists/Ankles)
        self.weights = np.array([
            0.8, 0.8,  # Elbows
            1.0, 1.0,  # Shoulders
            0.3, 0.3,  # Wrists (prone to noise, less critical to core form)
            1.2, 1.2,  # Hips (core)
            1.2, 1.2,  # Knees (foundation)
            0.3, 0.3,  # Ankles 
            1.5        # Spine (posture is paramount)
        ])
        self.weights = self.weights / np.sum(self.weights) # Normalize to sum=1

        # 2. Biological Maximum Range of Motion (Max possible error per joint in degrees)
        self.max_rom = np.array([
            150, 150,  # Elbows
            180, 180,  # Shoulders
            120, 120,  # Wrists
            160, 160,  # Hips
            160, 160,  # Knees
            90,  90,   # Ankles
            90         # Spine 
        ])

    def calculate_temporal_stability(self, series):
        """Calculate motion smoothness (mean absolute first derivative)."""
        if len(series) < 2: return np.zeros(series.shape[1])
        velocities = np.abs(np.diff(series, axis=0))
        return np.mean(velocities, axis=0)
        
    def weighted_dtw_distance(self, s1, s2):
        """Custom distance metric for FastDTW incorporating joint weights & ROM normalization."""
        # Normalize vectors by Biological ROM
        norm_s1, norm_s2 = s1 / self.max_rom, s2 / self.max_rom
        
        # Weighted Euclidean
        diff = norm_s1 - norm_s2
        weighted_sq_diff = np.sum((diff ** 2) * self.weights)
        return np.sqrt(weighted_sq_diff)
        
    def evaluate(self, student_data, teacher_data):
        try:
            # Ensure inputs are numpy arrays
            student_data = np.array(student_data)
            teacher_data = np.array(teacher_data)
            
            # Guard rails
            if len(student_data) == 0 or len(teacher_data) == 0:
                return {"overall_score": 0, "pose_score": 0, "jitter_penalty": 0, "regional": {"arms": 0, "legs": 0, "torso": 0}}

            # --- A. DTW Alignment with Biomechanical Weights ---
            # fastdtw returns (distance, path)
            distance, path = fastdtw(student_data, teacher_data, dist=self.weighted_dtw_distance)
            path_len = len(path)
            
            # Normalize distance (average weighted error across the path)
            # In our normalized scale, worst-case distance per frame is ~1.0
            avg_distance = (distance / path_len) if path_len > 0 else 1.0
            
            # Pose accuracy converts distance to a 0-100 score
            pose_score = max(0, min(100, (1 - avg_distance) * 100))
            
            # --- B. Temporal Stability Penalty ---
            # Calculate smoothness: if student exhibits massive jitter vs teacher
            student_smoothness = self.calculate_temporal_stability(student_data)
            teacher_smoothness = self.calculate_temporal_stability(teacher_data)
            
            # Compare weighted jitter
            jitter_diff = np.maximum(0, student_smoothness - teacher_smoothness) # only penalize excess jitter
            weighted_jitter = np.sum((jitter_diff / self.max_rom) * self.weights)
            
            jitter_penalty = min(15, weighted_jitter * 50) # Cap penalty at 15 points
            
            # --- C. Compute Regional Breakdown ---
            # Map out which columns belong to which region dynamically based on the path
            s_aligned = np.array([student_data[i] for i, j in path])
            t_aligned = np.array([teacher_data[j] for i, j in path])
            
            abs_errors = np.abs(s_aligned - t_aligned) / self.max_rom
            mean_errors = np.mean(abs_errors, axis=0) # Shape: (13,)
            
            # Convert regional errors to 100-scale
            def region_score(indices):
                err = np.mean([mean_errors[i] for i in indices])
                return max(0, min(100, (1 - err) * 100))

            regional_scores = {
                "arms": region_score([0, 1, 2, 3, 4, 5]),
                "legs": region_score([6, 7, 8, 9, 10, 11]),
                "torso": region_score([12])
            }

            # --- D. Final Calculation ---
            final_score = max(0, min(100, pose_score - jitter_penalty))
            
            return {
                "overall_score": round(final_score, 2),
                "pose_score": round(pose_score, 2),
                "jitter_penalty": round(jitter_penalty, 2),
                "regional": {k: round(v, 2) for k, v in regional_scores.items()}
            }
        except Exception:
            return {"overall_score": 0, "pose_score": 0, "jitter_penalty": 0, "regional": {"arms": 0, "legs": 0, "torso": 0}}


def generate_detailed_feedback(student_data, student_ts, teacher_data, teacher_ts=None, threshold_deg=25, top_n=None):
    """Generate time-stamped, per-joint feedback using DTW alignment.

    Returns a list of feedback entries ordered by magnitude.
    Each entry: {'joint': str, 'start_time': float, 'end_time': float,
                 'avg_diff': float, 'student_angle': float, 'teacher_angle': float, 'message': str}
    """
    joint_names = ['L_Elbow', 'R_Elbow', 'L_Shoulder', 'R_Shoulder',
                   'L_Wrist', 'R_Wrist',
                   'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee',
                   'L_Ankle', 'R_Ankle', 'Spine']

    try:
        distance, path = fastdtw(student_data, teacher_data, dist=euclidean)
    except Exception:
        return []

    # path is list of (i,j) pairs mapping student idx -> teacher idx
    # Group matched pairs by student index order
    path_sorted = sorted(path, key=lambda p: p[0])

    # For each joint, collect list of (student_time, diff, s_angle, t_angle)
    per_joint_events = {j: [] for j in range(student_data.shape[1])}

    for s_idx, t_idx in path_sorted:
        if s_idx < 0 or t_idx < 0 or s_idx >= len(student_ts) or t_idx >= len(teacher_data):
            continue
        s_time = float(student_ts[s_idx])
        t_time = float(teacher_ts[t_idx]) if (teacher_ts is not None and t_idx < len(teacher_ts)) else None
        s_row = student_data[s_idx]
        t_row = teacher_data[t_idx]
        diffs = t_row - s_row
        for j in range(len(diffs)):
            per_joint_events[j].append((s_time, float(diffs[j]), float(s_row[j]), float(t_row[j]), t_time))

    # Per-joint effective threshold multiplier.
    # Lower-body joints (hips/knees/ankles) are penalised by camera-angle
    # distortion and clothing occlusion, so raise their bar before flagging.
    # Wrists/ankles (fast-moving extremities) also get a slight lift.
    joint_threshold_mult = {
        4: 1.3, 5: 1.3,   # L/R Wrist  – fast-moving extremities
        6: 1.5, 7: 1.5,   # L/R Hip    – most affected by camera angle
        8: 1.4, 9: 1.4,   # L/R Knee   – clothing + perspective
        10: 1.5, 11: 1.5, # L/R Ankle  – foot/floor occlusion
    }

    feedback_list = []

    for j, events in per_joint_events.items():
        if not events:
            continue

        eff_threshold = threshold_deg * joint_threshold_mult.get(j, 1.0)

        # Find contiguous segments where abs(diff) > eff_threshold
        runs = []
        run = []
        prev_time = None
        for (t, diff, s_ang, tr_ang, t_time) in events:
            if abs(diff) >= eff_threshold:
                if run and prev_time is not None and t - prev_time > 0.6:
                    runs.append(run)
                    run = []
                run.append((t, diff, s_ang, tr_ang, t_time))
            else:
                if run:
                    runs.append(run)
                    run = []
            prev_time = t
        if run:
            runs.append(run)

        MIN_DURATION_SEC = 0.5

        for run in runs:
            times = [r[0] for r in run]
            start_t = min(times)
            end_t = max(times)

            # Skip micro-mistakes: must persist for at least MIN_DURATION_SEC
            if (end_t - start_t) < MIN_DURATION_SEC:
                continue

            diffs = [r[1] for r in run]
            s_angs = [r[2] for r in run]
            t_angs = [r[3] for r in run]
            t_times = [r[4] for r in run if r[4] is not None]
            avg_diff = float(sum(diffs) / len(diffs))
            avg_s = float(sum(s_angs) / len(s_angs))
            avg_t = float(sum(t_angs) / len(t_angs))
            teacher_time = float(sum(t_times) / len(t_times)) if t_times else None

            # Direction: positive avg_diff means teacher angle > student angle -> student should increase angle
            if avg_diff > 0:
                direction = f"Increase {joint_names[j]} by {abs(avg_diff):.1f}°"
            else:
                direction = f"Decrease {joint_names[j]} by {abs(avg_diff):.1f}°"

            message = f"{direction} (student {avg_s:.1f}°, teacher {avg_t:.1f}°) around {start_t:.1f}s"

            feedback_list.append({
                'joint': joint_names[j],
                'start_time': start_t,
                'end_time': end_t,
                'avg_diff': abs(avg_diff),
                'student_angle': avg_s,
                'teacher_angle': avg_t,
                'message': message,
                'teacher_time': teacher_time
            })

    # Return sorted by avg_diff descending (worst joints first); no cap by default
    feedback_list.sort(key=lambda x: x['avg_diff'], reverse=True)
    return feedback_list if top_n is None else feedback_list[:top_n]


def _friendly_message(entry):
    joint = entry.get('joint', '')
    start = entry.get('start_time', 0.0)
    avg_diff = entry.get('avg_diff', 0.0)
    s_ang = entry.get('student_angle', 0.0)
    t_ang = entry.get('teacher_angle', 0.0)

    # Clean up joint name for display
    joint_name = joint.lower().replace('_', ' ')

    # Core Logic: If Teacher Angle > Student Angle, student needs to INCREASE their angle
    increase = t_ang > s_ang

    if 'knee' in joint_name or 'elbow' in joint_name:
        verb = 'straighten' if increase else 'bend'
    elif 'shoulder' in joint_name:
        verb = 'raise' if increase else 'lower'
    elif 'hip' in joint_name:
        verb = 'stand straighter' if increase else 'bend deeper at the waist/hips'
    elif 'wrist' in joint_name:
        verb = 'straighten/extend' if increase else 'bend/flex'
    elif 'ankle' in joint_name:
        verb = 'point your toes more' if increase else 'flex your foot upward'
    elif 'spine' in joint_name or 'posture' in joint_name:
        verb = 'stand taller' if increase else 'lean forward'
    else:
        verb = 'increase angle of' if increase else 'decrease angle of'

    return f"At {start:.1f}s — {verb.capitalize()} your {joint_name} by about {abs(avg_diff):.0f}° (you: {s_ang:.0f}°, target: {t_ang:.0f}°)."


def get_semantic_feedback(angle_name, student_angle, teacher_angle):
    part = angle_name.lower().replace('_', ' ')
    
    # diff = Student - Teacher
    # If diff < 0 (Negative), Student is smaller than Teacher -> Needs to INCREASE angle
    # If diff > 0 (Positive), Student is larger than Teacher -> Needs to DECREASE angle
    diff = student_angle - teacher_angle
    abs_diff = abs(diff)

    if abs_diff < 10:
        return f"Perfect {part}!"

    if 'elbow' in part or 'knee' in part:
        if diff < -15: return f"Straighten your {part}."
        if diff > 15:  return f"Bend your {part} more."

    elif 'shoulder' in part:
        if diff < -15: return f"Raise your {part} higher."
        if diff > 15:  return f"Lower your {part}."

    elif 'hip' in part:
        if diff < -15: return f"Extend your hips / stand up straighter."
        if diff > 15:  return f"Bend deeper at the hips / squat lower."

    elif 'wrist' in part:
        if diff < -15: return f"Extend/straighten your {part} more."
        if diff > 15:  return f"Flex/bend your {part} more."

    elif 'ankle' in part:
        if diff < -15: return f"Point your toes more (extend {part})."
        if diff > 15:  return f"Flex your foot upward more."

    elif 'spine' in part or 'posture' in part:
        if diff < -15: return "Stand taller — straighten your back."
        if diff > 15:  return "Lean your torso forward more."

    # Fallback
    if diff < 0: return f"Open up / increase the angle of your {part}."
    else:        return f"Close / decrease the angle of your {part}."



def compose_side_by_side(teacher_path, student_path, out_path,
                         teacher_start=0.0, student_start=0.0, duration=None,
                         joint_label=None, fast_mode=False):
    """Produce a side-by-side annotated MP4 using OpenCV frame-by-frame rendering.

    For per-segment clips:
      • Pose skeleton is drawn on every frame (MediaPipe).
      • Problem joint bones/dots are highlighted RED on the student (right) side.
      • Teacher skeleton (left) is drawn in GREEN as a correct reference.
      • 'Teacher' / 'Student' labels and a red 'FIX <joint>' badge are burned in
        via cv2.putText — no FFmpeg drawtext dependency needed.

    Falls back to a plain FFmpeg compose if OpenCV writing fails.
    Returns True on success, False on error.
    """
    TARGET_H  = 480
    OUT_FPS   = 30
    highlight_ids = JOINT_TO_LANDMARK_IDS.get(joint_label or '', [])

    def _annotate_and_write() -> bool:
        t_cap = cv2.VideoCapture(teacher_path)
        s_cap = cv2.VideoCapture(student_path)
        if not t_cap.isOpened() or not s_cap.isOpened():
            t_cap.release(); s_cap.release()
            return False

        t_fps = t_cap.get(cv2.CAP_PROP_FPS) or OUT_FPS
        s_fps = s_cap.get(cv2.CAP_PROP_FPS) or OUT_FPS

        t_cap.set(cv2.CAP_PROP_POS_MSEC, teacher_start * 1000)
        s_cap.set(cv2.CAP_PROP_POS_MSEC, student_start  * 1000)

        max_frames = int((duration or 600) * OUT_FPS)

        # Probe frame size to set up VideoWriter
        ret_t, probe_t = t_cap.read()
        ret_s, probe_s = s_cap.read()
        if not ret_t and not ret_s:
            t_cap.release(); s_cap.release()
            return False

        # Seek back to start after probe
        t_cap.set(cv2.CAP_PROP_POS_MSEC, teacher_start * 1000)
        s_cap.set(cv2.CAP_PROP_POS_MSEC, student_start  * 1000)

        def _scale(frame, target_h):
            if frame is None:
                return None
            h, w = frame.shape[:2]
            scale = target_h / float(h)
            return cv2.resize(frame, (max(1, int(w * scale)), target_h))

        # Determine output width from probes
        pt = _scale(probe_t if ret_t else probe_s, TARGET_H)
        ps = _scale(probe_s if ret_s else probe_t, TARGET_H)
        out_w = (pt.shape[1] if pt is not None else TARGET_H) + \
                (ps.shape[1] if ps is not None else TARGET_H)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, float(OUT_FPS), (out_w, TARGET_H))
        if not writer.isOpened():
            t_cap.release(); s_cap.release()
            return False

        t_interval = max(1, round(t_fps / OUT_FPS))
        s_interval = max(1, round(s_fps / OUT_FPS))
        t_frame_buf = None
        s_frame_buf = None
        t_frame_idx = 0
        s_frame_idx = 0

        for frame_i in range(max_frames):
            # Read teacher frame at its native rate (sub-sample to OUT_FPS)
            if frame_i % t_interval == 0 or t_frame_buf is None:
                ret_t, raw_t = t_cap.read()
                if ret_t:
                    t_frame_buf = raw_t

            if frame_i % s_interval == 0 or s_frame_buf is None:
                ret_s, raw_s = s_cap.read()
                if ret_s:
                    s_frame_buf = raw_s

            if t_frame_buf is None and s_frame_buf is None:
                break

            # Draw pose skeleton on each frame
            t_ann = draw_pose_on_frame(_scale(t_frame_buf, TARGET_H),
                                       highlight_landmark_ids=[])   # green reference
            s_ann = draw_pose_on_frame(_scale(s_frame_buf, TARGET_H),
                                       highlight_landmark_ids=highlight_ids)  # red highlights

            # Fallback: plain scaled frames if annotation returned None
            if t_ann is None: t_ann = _scale(t_frame_buf, TARGET_H)
            if s_ann is None: s_ann = _scale(s_frame_buf, TARGET_H)

            # If one side ran out of frames, use a black placeholder
            if t_ann is None:
                tw = ps.shape[1] if ps is not None else TARGET_H
                t_ann = np.zeros((TARGET_H, tw, 3), dtype=np.uint8)
            if s_ann is None:
                sw = pt.shape[1] if pt is not None else TARGET_H
                s_ann = np.zeros((TARGET_H, sw, 3), dtype=np.uint8)

            # Burn-in text labels
            font      = cv2.FONT_HERSHEY_SIMPLEX
            lbl_scale = 0.65
            lbl_thick = 2

            def _label_box(img, text, color_bg, color_fg, x, y):
                (tw, th), bl = cv2.getTextSize(text, font, lbl_scale, lbl_thick)
                cv2.rectangle(img, (x, y - th - 4), (x + tw + 8, y + bl), color_bg, -1)
                cv2.putText(img, text, (x + 4, y), font, lbl_scale, color_fg, lbl_thick, cv2.LINE_AA)

            _label_box(t_ann, 'Teacher', (0, 0, 0),   (255, 255, 255), 6, t_ann.shape[0] - 10)
            _label_box(s_ann, 'Student', (0, 0, 0),   (255, 255, 255), 6, s_ann.shape[0] - 10)

            if joint_label:
                fix_text = f'FIX: {joint_label}'
                _label_box(s_ann, fix_text, (0, 30, 220), (255, 255, 255), 6, 30)

            # Pad/crop to expected widths so hconcat never fails
            def _ensure_w(img, want_w):
                h, w = img.shape[:2]
                if w == want_w:
                    return img
                if w > want_w:
                    return img[:, :want_w]
                pad = np.zeros((h, want_w - w, 3), dtype=np.uint8)
                return np.hstack([img, pad])

            t_w = out_w // 2
            s_w = out_w - t_w
            combined = np.hstack([_ensure_w(t_ann, t_w), _ensure_w(s_ann, s_w)])
            writer.write(combined)

        writer.release()
        t_cap.release()
        s_cap.release()
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0

    if not fast_mode:
        try:
            if _annotate_and_write():
                # Re-mux with FFmpeg to ensure proper H.264 in a browser-compatible mp4
                tmp_path = out_path + '.tmp.mp4'
                try:
                    res = subprocess.run(
                        ['ffmpeg', '-y', '-i', out_path,
                         '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26',
                         '-movflags', '+faststart', tmp_path],
                        capture_output=True, timeout=120
                    )
                    if res.returncode == 0:
                        os.replace(tmp_path, out_path)
                except Exception:
                    pass  # keep the raw mp4v file; most modern browsers support it
                return True
        except Exception as e:
            print(f"[compose] annotated write error: {e}")

    # ── Plain FFmpeg fallback (no skeleton) ──────────────────────────────────
    try:
        dur_flag = ['-t', str(duration)] if duration else []
        filter_g = (
            f"[0:v]trim=start={teacher_start},setpts=PTS-STARTPTS,fps=30,scale=-2:480,setsar=1,format=yuv420p[tv];"
            f"[1:v]trim=start={student_start},setpts=PTS-STARTPTS,fps=30,scale=-2:480,setsar=1,format=yuv420p[sv];"
            "[tv][sv]hstack=inputs=2[out]"
        )
        cmd = (['ffmpeg', '-y',
                '-i', teacher_path,
                '-i', student_path]
               + dur_flag
               + ['-filter_complex', filter_g,
                  '-map', '[out]',
                  '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26',
                  '-movflags', '+faststart', '-an', out_path])
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            print("[compose] FFmpeg Failed:", result.stderr.decode('utf-8', errors='replace'))
        return result.returncode == 0
    except Exception as e:
        print(f"[compose] FFmpeg fallback error: {e}")
        return False


def save_feedback_thumbnails(student_video_path, detailed_feedback, student_name, teacher_video_path=None):
    """Create side-by-side collage thumbnails for each feedback entry.
    If `teacher_video_path` is provided and a teacher_time exists in entry,
    capture both frames and combine horizontally. Attach `thumbnail` URL and
    friendly `message` to each entry and return updated list.
    """
    out_dir = os.path.join(app.static_folder, 'uploads_thumbs')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    student_cap = cv2.VideoCapture(student_video_path)
    teacher_cap = cv2.VideoCapture(teacher_video_path) if teacher_video_path else None

    updated = []
    for idx, entry in enumerate(detailed_feedback):
        s_time = float(entry.get('start_time', 0.0))
        t_time = entry.get('teacher_time', None)
        filename = secure_filename(f"{student_name}_{int(s_time*1000)}_{idx}.jpg")
        out_path = os.path.join(out_dir, filename)
        url_path = f"/static/uploads_thumbs/{filename}"

        s_frame = None
        t_frame = None

        try:
            # student frame
            student_cap.set(cv2.CAP_PROP_POS_MSEC, int(s_time * 1000))
            ret_s, s_frame = student_cap.read()
            if not ret_s:
                s_frame = None

            # teacher frame, if available
            if teacher_cap is not None and t_time is not None:
                teacher_cap.set(cv2.CAP_PROP_POS_MSEC, int(float(t_time) * 1000))
                ret_t, t_frame = teacher_cap.read()
                if not ret_t:
                    t_frame = None

            # Build collage: prefer side-by-side (teacher on left, student on right)
            if t_frame is not None and s_frame is not None:
                # resize both to same height
                th, tw = t_frame.shape[:2]
                sh, sw = s_frame.shape[:2]
                target_h = 240
                t_scale = target_h / float(th)
                s_scale = target_h / float(sh)
                t_resized = cv2.resize(t_frame, (int(tw * t_scale), target_h))
                s_resized = cv2.resize(s_frame, (int(sw * s_scale), target_h))
                collage = cv2.hconcat([t_resized, s_resized])
                cv2.imwrite(out_path, collage)
                entry['thumbnail'] = url_path
            elif s_frame is not None:
                # single student thumbnail
                sh, sw = s_frame.shape[:2]
                target_w = 480
                scale = target_w / float(sw)
                thumb = cv2.resize(s_frame, (target_w, int(sh * scale)))
                cv2.imwrite(out_path, thumb)
                entry['thumbnail'] = url_path
            elif t_frame is not None:
                th, tw = t_frame.shape[:2]
                target_w = 480
                scale = target_w / float(tw)
                thumb = cv2.resize(t_frame, (target_w, int(th * scale)))
                cv2.imwrite(out_path, thumb)
                entry['thumbnail'] = url_path
            else:
                entry['thumbnail'] = None
        except Exception:
            entry['thumbnail'] = None

        # Add friendly message text and semantic dancer-language feedback
        entry['message'] = _friendly_message(entry)
        try:
            entry['semantic'] = get_semantic_feedback(entry.get('joint',''), entry.get('student_angle',0.0), entry.get('teacher_angle',0.0))
        except Exception:
            entry['semantic'] = entry.get('message')
        updated.append(entry)

    student_cap.release()
    if teacher_cap is not None:
        teacher_cap.release()

    return updated

@app.route('/')
def index():
    """Main page"""
    print("Serving index page")
    return render_template('index.html')

@app.route('/test')
def test():
    """Test route"""
    return {'status': 'OK', 'message': 'Flask is running!'}

@app.route('/api/upload-teacher', methods=['POST'])
def upload_teacher():
    """Handle teacher video upload and reference creation"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use MP4, AVI, MOV, MKV, or WebM'}), 400
    
    try:
        # Save temporary file
        # Save teacher reference video permanently to uploads/teacher_reference.mp4
        filename = 'teacher_reference.mp4'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract angles and timestamps from saved teacher video
        dance_data, timestamps, landmarks_data, error = extract_angles_from_video(filepath)

        if error:
            os.remove(filepath)
            return jsonify({'error': error}), 400
        
        # Save to CSV
        with open('dance_data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['L_Elbow', 'R_Elbow', 'L_Shoulder', 'R_Shoulder',
                           'L_Wrist', 'R_Wrist',
                           'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee',
                           'L_Ankle', 'R_Ankle', 'Spine'])
            writer.writerows(dance_data)

        # Save teacher landmarks
        import json
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'teacher_landmarks.json'), 'w') as lf:
            json.dump(landmarks_data, lf)

        # Save teacher timestamps for later thumbnail extraction
        ts_path = os.path.join(app.config['UPLOAD_FOLDER'], 'teacher_timestamps.npy')
        try:
            np.save(ts_path, timestamps)
        except Exception:
            pass

        # Copy teacher video into static folder for in-page playback
        try:
            static_videos = os.path.join(app.static_folder, 'uploads_videos')
            os.makedirs(static_videos, exist_ok=True)
            teacher_static_path = os.path.join(static_videos, 'teacher_reference.mp4')
            shutil.copy2(filepath, teacher_static_path)
            shutil.copy2(os.path.join(app.config['UPLOAD_FOLDER'], 'teacher_landmarks.json'), 
                         os.path.join(static_videos, 'teacher_landmarks.json'))
        except Exception:
            teacher_static_path = None
        # Keep original teacher file in uploads as well
        
        return jsonify({
            'success': True,
            'message': f'Teacher reference created successfully!',
            'frames': len(dance_data)
        })
    
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

@app.route('/api/upload-student', methods=['POST'])
def upload_student():
    """Handle student video upload and grading"""
    # Check if teacher reference exists (data and timestamps)
    teacher_data, teacher_ts, error = load_teacher_data()
    if error:
        return jsonify({'error': 'Please upload teacher video first'}), 400
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save temporary file
        student_name = request.form.get('name', 'Student')
        filename = secure_filename(f"student_{student_name}_{datetime.now().timestamp()}.mp4")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract angles from video (also get timestamps)
        student_data, student_ts, landmarks_data, error = extract_angles_from_video(filepath)

        if error:
            os.remove(filepath)
            return jsonify({'error': error}), 400

        # ── Audio-based temporal alignment ──────────────────────────────────
        # Find where the two recordings are relative to the same piece of music
        # and trim the pose arrays so both start at the same musical beat.
        teacher_video_path_raw = os.path.join(app.config['UPLOAD_FOLDER'], 'teacher_reference.mp4')
        audio_offset = 0.0
        t_trim_sec   = 0.0
        s_trim_sec   = 0.0
        if os.path.exists(teacher_video_path_raw):
            try:
                audio_offset = find_audio_offset(teacher_video_path_raw, filepath)
                teacher_data, teacher_ts, student_data, student_ts, t_trim_sec, s_trim_sec = \
                    align_pose_data(teacher_data, teacher_ts, student_data, student_ts, audio_offset)
                print(f'[align] t_trim={t_trim_sec:.3f}s  s_trim={s_trim_sec:.3f}s  '
                      f'→ teacher_frames={len(teacher_data)}  student_frames={len(student_data)}')
            except Exception as ae:
                print(f'[align] Alignment skipped: {ae}')
        # ────────────────────────────────────────────────────────────────────
        
        # Save student data to CSV
        csv_filename = f"student_{student_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
        
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['L_Elbow', 'R_Elbow', 'L_Shoulder', 'R_Shoulder',
                           'L_Wrist', 'R_Wrist',
                           'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee',
                           'L_Ankle', 'R_Ankle', 'Spine'])
            writer.writerows(student_data)

        # Also save student landmarks
        import json
        with open(os.path.join(app.config['UPLOAD_FOLDER'], f"student_{student_name}_landmarks.json"), 'w') as f:
            json.dump(landmarks_data, f)

        # Copy student video into static folder for in-page playback (persistent)
        try:
            static_videos = os.path.join(app.static_folder, 'uploads_videos')
            os.makedirs(static_videos, exist_ok=True)
            student_static_name = f"student_{student_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            student_static_path = os.path.join(static_videos, student_static_name)
            shutil.copy2(filepath, student_static_path)
            shutil.copy2(os.path.join(app.config['UPLOAD_FOLDER'], f"student_{student_name}_landmarks.json"), 
                         os.path.join(static_videos, f"student_{student_name}_landmarks.json"))
            student_video_url = f"/static/uploads_videos/{student_static_name}"
        except Exception:
            student_video_url = None
        
        # Calculate scores using the new BiomechanicalScorer
        scorer = BiomechanicalScorer()
        scoring_results = scorer.evaluate(student_data, teacher_data)
        
        overall_score = scoring_results['overall_score']
        arm_score = scoring_results['regional']['arms']
        leg_score = scoring_results['regional']['legs']
        torso_score = scoring_results['regional']['torso']

        # Generate detailed feedback (time-stamped)
        detailed = generate_detailed_feedback(student_data, student_ts, teacher_data, teacher_ts=teacher_ts, threshold_deg=12)

        # Capture thumbnails for each feedback item and add friendly messages
        teacher_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'teacher_reference.mp4')
        if os.path.exists(teacher_video_path):
            detailed = save_feedback_thumbnails(filepath, detailed, student_name, teacher_video_path=teacher_video_path)
        else:
            detailed = save_feedback_thumbnails(filepath, detailed, student_name, teacher_video_path=None)

        # Render the aligned side-by-side video on the server
        composed_name = f"composed_{student_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        composed_path = os.path.join(app.static_folder, 'uploads_videos', composed_name)
        
        teacher_video_raw = os.path.join(app.config['UPLOAD_FOLDER'], 'teacher_reference.mp4')
        if os.path.exists(teacher_video_raw):
            success = compose_side_by_side(
                teacher_video_raw, filepath, composed_path,
                teacher_start=t_trim_sec, student_start=s_trim_sec, 
                fast_mode=True
            )
            if success:
                composed_full_url = f"/static/uploads_videos/{composed_name}"
            else:
                composed_full_url = None
        else:
            composed_full_url = None

        
        # Determine feedback
        if overall_score >= 80:
            feedback = "Excellent! Your movements match the teacher very closely!"
            star_rating = 5
        elif overall_score >= 70:
            feedback = "Good job! Minor adjustments needed in form and timing."
            star_rating = 4
        elif overall_score >= 60:
            feedback = "Fair attempt. Practice more to improve your technique."
            star_rating = 3
        else:
            feedback = "Keep practicing! Focus on body alignment and joint angles."
            star_rating = 2
        
        # Save results (attach detailed feedback before persisting)
        results = {
            'success': True,
            'student_name': student_name,
            'frames': len(student_data),
            'overall_score': overall_score,
            'arm_score': arm_score,
            'leg_score': leg_score,
            'torso_score': torso_score,
            'feedback': feedback,
            'star_rating': star_rating,
            'timestamp': datetime.now().isoformat(),
            'composed_video': composed_full_url,
            'audio_offset_sec': round(audio_offset, 3),
            'teacher_trim_sec': round(t_trim_sec, 3),
            'student_trim_sec': round(s_trim_sec, 3)
        }
        
        # Append to results file
        results_file = os.path.join(app.config['UPLOAD_FOLDER'], 'grading_results.json')
        grading_history = []
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                grading_history = json.load(f)
        
        # Attach detailed feedback to results
        results['detailed_feedback'] = detailed
        # Include URLs for in-page synchronized playback (teacher + student)
        try:
            teacher_static_url = '/static/uploads_videos/teacher_reference.mp4' if os.path.exists(os.path.join(app.static_folder, 'uploads_videos', 'teacher_reference.mp4')) else None
        except Exception:
            teacher_static_url = None
        results['teacher_video'] = teacher_static_url
        results['student_video'] = student_video_url if 'student_video_url' not in locals() else student_video_url
        results['teacher_landmarks_url'] = "/static/uploads_videos/teacher_landmarks.json"
        results['student_landmarks_url'] = f"/static/uploads_videos/student_{student_name}_landmarks.json"

        # Produce prioritized semantic feedback for Arms, Legs, Torso
        try:
            # Arms cols 0-5, Legs cols 6-11, Torso col 12
            min_len = min(len(student_data), len(teacher_data))
            diffs = np.abs(student_data[:min_len] - teacher_data[:min_len])
            mean_diffs = np.nanmean(diffs, axis=0)
            # arms (0-5)
            arm_idx = int(np.argmax(mean_diffs[0:6]))
            arm_label_map = ['L_Elbow', 'R_Elbow', 'L_Shoulder', 'R_Shoulder', 'L_Wrist', 'R_Wrist']
            arm_msg = get_semantic_feedback(arm_label_map[arm_idx],
                float(np.mean(student_data[:, arm_idx])), float(np.mean(teacher_data[:, arm_idx])))
            # legs (6-11)
            leg_offset = int(6 + np.argmax(mean_diffs[6:12]))
            leg_label_map = ['L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle']
            leg_msg = get_semantic_feedback(leg_label_map[leg_offset - 6],
                float(np.mean(student_data[:, leg_offset])), float(np.mean(teacher_data[:, leg_offset])))
            # torso (12)
            torso_msg = get_semantic_feedback('Spine',
                float(np.mean(student_data[:, 12])), float(np.mean(teacher_data[:, 12])))

            results['semantic_feedback'] = {'arm': arm_msg, 'leg': leg_msg, 'torso': torso_msg}
        except Exception:
            results['semantic_feedback'] = {'arm': '', 'leg': '', 'torso': ''}

        grading_history.append(results)

        with open(results_file, 'w') as f:
            json.dump(grading_history, f, indent=2)

        # Clean up
        os.remove(filepath)

        return jsonify(results)
    
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

@app.route('/api/history')
def get_history():
    """Get grading history"""
    results_file = os.path.join(app.config['UPLOAD_FOLDER'], 'grading_results.json')
    
    if not os.path.exists(results_file):
        return jsonify([])
    
    try:
        with open(results_file, 'r') as f:
            history = json.load(f)
        return jsonify(history)
    except:
        return jsonify([])

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear all grading history"""
    results_file = os.path.join(app.config['UPLOAD_FOLDER'], 'grading_results.json')

    cleanup_patterns = {
        app.config['UPLOAD_FOLDER']: [
            'student_*.csv',
            'student_*.mp4',
        ],
        os.path.join(app.static_folder, 'uploads_videos'): [
            'student_*.mp4',
        ],
        os.path.join(app.static_folder, 'uploads_clips'): [
            'clip_*.mp4',
            'full_*.mp4',
        ],
        os.path.join(app.static_folder, 'uploads_thumbs'): [
            '*.jpg',
            '*.jpeg',
            '*.png',
        ],
    }

    try:
        deleted_count = 0

        if os.path.exists(results_file):
            os.remove(results_file)

        for base_dir, patterns in cleanup_patterns.items():
            if not os.path.exists(base_dir):
                continue

            for pattern in patterns:
                for target_path in glob.glob(os.path.join(base_dir, pattern)):
                    if os.path.isfile(target_path):
                        os.remove(target_path)
                        deleted_count += 1

        return jsonify({'success': True, 'deleted_files': deleted_count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Get system status"""
    teacher_exists = os.path.exists('dance_data.csv')
    teacher_frames = 0
    
    if teacher_exists:
        with open('dance_data.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            teacher_frames = sum(1 for row in reader)
    
    results_file = os.path.join(app.config['UPLOAD_FOLDER'], 'grading_results.json')
    student_count = 0
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            student_count = len(json.load(f))
    
    return jsonify({
        'teacher_exists': teacher_exists,
        'teacher_frames': teacher_frames,
        'students_graded': student_count
    })

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)
