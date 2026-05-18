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
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import LoginManager, login_required, current_user
from werkzeug.utils import secure_filename
from models.database import db
from models.user import User
from models.attempt import Attempt
from auth.routes import auth_bp
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import correlate, savgol_filter, find_peaks
from scipy.io import wavfile
from datetime import datetime
import json
import shutil
from flask.json.provider import DefaultJSONProvider
from flask import send_from_directory
from reference_loader import (load_all_references, get_reference,
                               has_reference, list_available_references,
                               get_all_references_meta)

class NumpyJSONProvider(DefaultJSONProvider):
    @staticmethod
    def default(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return DefaultJSONProvider.default(obj)

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['WTF_CSRF_ENABLED'] = True
# PENTING: ganti secret_key dengan random string di production
app.secret_key = 'dance-grading-secret-CHANGE-IN-PRODUCTION'

# Init extensions
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'auth.login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register blueprint
app.register_blueprint(auth_bp)

# Create tables on first run
with app.app_context():
    db.create_all()

# Load reference pose cache into memory
load_all_references()

# Debugging
print(f"✓ Template folder: {os.path.abspath(app.template_folder)}")
print(f"✓ Static folder: {os.path.abspath(app.static_folder)}")

# MediaPipe setup
mp_pose = mp.solutions.pose

# =====================================================================
# EXPERT JUDGMENT CONFIGURATION (Ground Truth)
# =====================================================================
MOVEMENT_CONFIG = {
    'bouncing': {
        'core_joints': [8, 9],   # Knees only
        'weights': {'wiraga': 0.50, 'wirama': 0.50, 'power': 0.0},
        'thresholds': {'knee_bend_target': 90.0}
    },
    'stepping': {
        'core_joints': [10, 11],   # Ankles
        'weights': {'wiraga': 0.50, 'wirama': 0.50, 'power': 0.0},
        'thresholds': {'timing_tolerance': 0.1}
    },
    'sliding': {
        'core_joints': [6, 7, 10, 11, 12],   # Hips + Ankles + Spine
        'weights': {'wiraga': 0.50, 'wirama': 0.50, 'power': 0.0},
        'thresholds': {'sliding_dist_shoulder_ratio': 1.0}
    },
}

# =====================================================================
# ONE EURO FILTER — real-time per-joint jitter removal
# Applied inside extract_angles_from_video before Savitzky-Golay.
# Adaptive cutoff: slow movements get heavy smoothing, fast hits stay sharp.
# Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter"
# =====================================================================
import math

class OneEuroFilter:
    """Adaptive low-pass filter for 1-D or N-D signals.

    Parameters
    ----------
    t0          : float  — initial timestamp (seconds)
    x0          : array  — initial value (scalar or numpy array)
    min_cutoff  : float  — minimum cutoff frequency (Hz).  Lower = smoother at rest.
    beta        : float  — speed coefficient.  Higher = less lag during fast moves.
    d_cutoff    : float  — cutoff for the derivative (fixed).
    """
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=0.5, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta       = float(beta)
        self.d_cutoff   = float(d_cutoff)
        self.x_prev     = np.array(x0, dtype=float)
        self.dx_prev    = np.array(dx0 if np.ndim(dx0) > 0 else
                                   np.zeros_like(self.x_prev), dtype=float)
        self.t_prev     = float(t0)

    def _alpha(self, t_e, cutoff):
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def __call__(self, t, x):
        x = np.array(x, dtype=float)
        t_e = t - self.t_prev
        if t_e <= 0:
            return x

        a_d    = self._alpha(t_e, self.d_cutoff)
        dx     = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a      = self._alpha(t_e, cutoff)
        x_hat  = a * x + (1.0 - a) * self.x_prev

        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
        return x_hat

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
    smoothed_data = savgol_filter(pose_data, window_length=window_length, polyorder=polyorder, axis=0)
    
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

def downsample_pose_data(pose_data, timestamps, target_fps=12.0):
    """Downsample pose and timestamp arrays to reduce FastDTW compute load.
    
    Keeps motion pattern intact while reducing frame count.
    E.g., 30fps → 12fps removes 60% of frames, speeds DTW by ~6x.
    """
    if len(pose_data) < 2 or len(timestamps) < 2:
        return pose_data, timestamps
    
    current_fps = len(pose_data) / float(timestamps[-1]) if timestamps[-1] > 0 else 30.0
    if current_fps <= target_fps:
        return pose_data, timestamps  # already slow enough
    
    stride = max(1, int(round(current_fps / target_fps)))
    return pose_data[::stride], timestamps[::stride]

def extract_angles_from_video(video_path):
    """Extract dance angles from video and return per-frame timestamps.

    Returns (dance_data_array, timestamps_array, landmarks_data, error_message) on success or
    (None, None, None, error_message) on failure.

    Visibility gating: each angle is only calculated when all three of its
    constituent MediaPipe landmarks have visibility >= VIS_THRESHOLD.  When any
    landmark is occluded / uncertain the angle falls back to the most recent
    reliable value so the pose array stays dense (no NaN gaps).
    
    ERROR HANDLING: If no pose landmarks are detected in a frame, that frame is
    silently skipped. If the entire video has < 5 valid pose frames, an error
    is returned rather than crashing.
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

    # OneEuroFilter state — one filter per MediaPipe landmark (for x,y,z coords)
    # min_cutoff=0.5 Hz keeps slow grooves smooth; beta=0.01 preserves sharp hits
    _euro_filters: dict = {}

    def get_smoothed_xyz(lm, lm_id_enum, current_time):
        """Return 1€-filtered [x, y, z] for a single landmark."""
        raw_xyz = np.array([lm[lm_id_enum.value].x,
                            lm[lm_id_enum.value].y,
                            lm[lm_id_enum.value].z], dtype=float)
        key = lm_id_enum.value
        if key not in _euro_filters:
            _euro_filters[key] = OneEuroFilter(t0=current_time, x0=raw_xyz,
                                               min_cutoff=0.5, beta=0.01)
            return raw_xyz
        return _euro_filters[key](t=current_time, x=raw_xyz)

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
         try:
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

             # ── PERBAIKAN 2: Strict error handling for empty frames ──
             if results is None or results.pose_landmarks is None:
                 # No pose detected in this frame — skip silently
                 continue

             lm = results.pose_landmarks.landmark
             PL = mp_pose.PoseLandmark

             # Grab 1€-filtered [x,y,z] coordinates — jitter removed per joint
             current_time = raw_frame_index / float(fps)
             def fxy(lm_id_enum):
                 return get_smoothed_xyz(lm, lm_id_enum, current_time).tolist()

             l_sh = fxy(PL.LEFT_SHOULDER);   r_sh = fxy(PL.RIGHT_SHOULDER)
             l_el = fxy(PL.LEFT_ELBOW);      r_el = fxy(PL.RIGHT_ELBOW)
             l_wr = fxy(PL.LEFT_WRIST);      r_wr = fxy(PL.RIGHT_WRIST)
             l_hi = fxy(PL.LEFT_HIP);        r_hi = fxy(PL.RIGHT_HIP)
             l_kn = fxy(PL.LEFT_KNEE);       r_kn = fxy(PL.RIGHT_KNEE)
             l_an = fxy(PL.LEFT_ANKLE);      r_an = fxy(PL.RIGHT_ANKLE)
             l_idx = fxy(PL.LEFT_INDEX);     r_idx = fxy(PL.RIGHT_INDEX)
             l_ft  = fxy(PL.LEFT_FOOT_INDEX); r_ft = fxy(PL.RIGHT_FOOT_INDEX)
             nose  = fxy(PL.NOSE)
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

         except Exception as frame_error:
             # Per-frame processing error — log and continue without crashing
             # This protects against frames with corrupt data, edge cases, etc.
             print(f"[pose] Frame {raw_frame_index} error: {frame_error}")
             pass

    cap.release()

    if not dance_data:
     return None, None, None, "No pose detected in video"

    # Smooth the extracted angle data using the Savitzky-Golay offline filter
    smoothed_data = smooth_pose_data(np.array(dance_data))

    return smoothed_data, np.array(timestamps), landmarks_data, None


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


def trim_leading_idle(pose_data, timestamps, vel, percentile=20, max_trim_sec=10.0):
    """Return seconds to skip at the start while dancer is standing still.

    Scans the velocity envelope from frame 0 and stops as soon as a rolling
    window exceeds the Nth-percentile threshold of the whole sequence.
    Returns 0.0 if no idle period is detected or data is too short.
    """
    if vel is None or len(vel) < 5 or len(timestamps) < 5:
        return 0.0
    envelope = np.mean(np.abs(vel), axis=1)
    threshold = np.percentile(envelope, percentile)
    fps = len(timestamps) / float(timestamps[-1]) if timestamps[-1] > 0 else 30.0
    window = max(3, int(fps))
    max_frames = int(fps * max_trim_sec)
    for i in range(min(max_frames, len(envelope) - window)):
        if np.mean(envelope[i:i + window]) > threshold:
            return float(timestamps[i]) if i > 0 else 0.0
    return 0.0


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


def extract_beat_times(video_path: str, sr: int = 22050) -> np.ndarray:
    """Extract beat timestamps (in seconds) from a video's audio track using onset detection.

    Strategy:
      1. Extract mono audio via ffmpeg.
      2. Compute a log-power onset strength envelope.
      3. Use dynamic-threshold peak picking to find beat/hit moments.

    Returns an array of beat times in seconds, or empty array on failure.
    """
    tmp_wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp_wav = f.name

        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(sr), '-ac', '1',
            '-t', '300',  # cap at 5 min
            tmp_wav
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=60)
        if res.returncode != 0:
            return np.array([])

        _, sig = wavfile.read(tmp_wav)
        sig = sig.astype(np.float32) / (np.iinfo(np.int16).max + 1)

        # Frame-based onset envelope: log RMS energy per 512-sample hop
        frame_len = 1024
        hop = 512
        n_frames = (len(sig) - frame_len) // hop
        if n_frames < 4:
            return np.array([])

        envelope = np.array([
            np.sqrt(np.mean(sig[i*hop: i*hop + frame_len] ** 2))
            for i in range(n_frames)
        ])
        # Log-compress to make soft and loud beats comparable
        envelope = np.log1p(envelope * 100)

        # First-order difference (onset strength)
        onset_strength = np.maximum(0, np.diff(envelope))

        # Dynamic threshold: median + 1.5 * std within a 1-second rolling window
        frame_sr = sr / hop   # frames per second
        win = max(1, int(frame_sr))
        thresholds = np.array([
            np.median(onset_strength[max(0, i-win): i+win]) +
            1.5 * (onset_strength[max(0, i-win): i+win].std() + 1e-6)
            for i in range(len(onset_strength))
        ])

        # Pick peaks above threshold with minimum 0.15 s separation
        min_sep = max(1, int(0.15 * frame_sr))
        beat_frames = []
        last = -min_sep
        for i in range(len(onset_strength)):
            if onset_strength[i] > thresholds[i] and (i - last) >= min_sep:
                beat_frames.append(i)
                last = i

        beat_times = np.array(beat_frames) * (hop / sr)
        return beat_times

    except Exception as e:
        print(f'[beats] extract_beat_times error: {e}')
        return np.array([])
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass


def compute_joint_kinematics(pose_data: np.ndarray, timestamps: np.ndarray):
    """Derive velocity and acceleration arrays from pose angle data.

    velocity[t]     = |angle[t+1] - angle[t]| / dt   (deg/s per joint)
    acceleration[t] = |velocity[t+1] - velocity[t]| / dt  (deg/s² per joint)

    Returns (velocity, acceleration) both shape (N-2, 13).
    If sequence is too short returns (None, None).
    """
    if len(pose_data) < 3 or len(timestamps) < 3:
        return None, None

    dt = np.diff(timestamps)
    dt = np.where(dt < 1e-6, 1e-6, dt)  # guard against zero-dt

    # Angular velocity: shape (N-1, 13)
    velocity = np.abs(np.diff(pose_data, axis=0)) / dt[:, np.newaxis]

    # Angular acceleration: shape (N-2, 13)
    dt2 = dt[:-1]
    acceleration = np.abs(np.diff(velocity, axis=0)) / dt2[:, np.newaxis]

    return velocity, acceleration


class MoveClassifier:
    """
    Classifies each frame (or segment) of a pose sequence into a Hip Hop
    move archetype, then returns a joint-weight mask so only the relevant
    joints are scored for that move.

    Joint indices (13 total):
      0  L_Elbow   1  R_Elbow
      2  L_Shoulder 3 R_Shoulder
      4  L_Wrist   5  R_Wrist
      6  L_Hip     7  R_Hip
      8  L_Knee    9  R_Knee
      10 L_Ankle   11 R_Ankle
      12 Spine

    Move archetypes (Coach Ambrosius Robby validation):
      bouncing    — vertical knee/hip dominant
      stepping    — ankle-led foot placement
      sliding     — hip + ankle horizontal travel
      full_body   — catch-all fallback (internal only)
    """

    # mask values: 1.0 = scored normally, 0.05 = nearly ignored, 0.0 = skip entirely
    MOVE_MASKS: dict = {
        # idx:        0     1     2     3     4     5     6     7     8     9    10    11    12
        #          L_Elb R_Elb L_Sho R_Sho L_Wri R_Wri L_Hip R_Hip L_Kne R_Kne L_Ank R_Ank Spin
        'bouncing':  [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40, 0.40, 1.50, 1.50, 0.05, 0.05, 0.30],
        'stepping':  [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.30, 0.30, 1.50, 1.50, 0.20],
        'sliding':   [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 0.30, 0.30, 1.00, 1.00, 1.30],
        'full_body': [0.70, 0.70, 1.00, 1.00, 0.30, 0.30, 1.20, 1.20, 1.10, 1.10, 0.30, 0.30, 1.50],
    }

    # ── Per-move scoring profiles ─────────────────────────────────────────────
    # Each profile defines:
    #   pillar_weights  — how much Timing / Movement / Power contribute to overall score
    #   focus_joints    — human-readable list of what's being judged (for UI)
    #   display_name    — friendly label shown to the user
    #   description     — one-line explanation of what the move emphasises
    #   feedback_cues   — what good execution looks and feels like
    MOVE_PROFILES: dict = {
        'bouncing': {
            'display_name': 'Bouncing',
            'description':  'Rhythmic vertical knee bounce on every beat',
            'pillar_weights': {'timing': 0.50, 'movement': 0.50, 'power': 0.0},
            'focus_joints':  ['Knees'],
            'ignored_joints': ['Arms', 'Wrists', 'Ankles'],
            'feedback_cues': 'Knees absorb and rebound on every beat. Hips drive the bounce — spine follows naturally.',
        },
        'stepping': {
            'display_name': 'Stepping',
            'description':  'Sharp foot-led steps with precise ankle placement and timing',
            'pillar_weights': {'timing': 0.50, 'movement': 0.50, 'power': 0.0},
            'focus_joints':  ['Ankles'],
            'ignored_joints': ['Arms', 'Spine'],
            'feedback_cues': 'Foot placement lands cleanly on the beat. No dragging.',
        },
        'sliding': {
            'display_name': 'Sliding',
            'description':  'Smooth horizontal travel — balance and body synchronisation are key',
            'pillar_weights': {'timing': 0.50, 'movement': 0.50, 'power': 0.0},
            'focus_joints':  ['Hips', 'Ankles', 'Spine (balance)'],
            'ignored_joints': ['Arms', 'Knees'],
            'feedback_cues': 'Smooth horizontal travel. Continuous without stiffness.',
        },
        'full_body': {
            'display_name': 'Full Body',
            'description':  'Internal fallback — all joints scored by move context',
            'pillar_weights': {'timing': 0.50, 'movement': 0.50, 'power': 0.0},
            'focus_joints':  ['All joints'],
            'ignored_joints': [],
            'feedback_cues': 'Move classifier adapts weights automatically per section.',
        },
    }

    # Normalised masks (each row sums to ~1 so scoring stays on same scale)
    _NORM_MASKS: dict = {}

    def __init__(self):
        for name, raw in self.MOVE_MASKS.items():
            arr = np.array(raw, dtype=float)
            total = arr.sum()
            self._NORM_MASKS[name] = arr / total if total > 1e-9 else arr

    # ── Heuristic classifier ───────────────────────────────────────────────────

    def classify_segment(self, vel_segment: np.ndarray) -> str:
        """
        Classify a short velocity segment (shape N×13) into a move archetype.
        """
        if vel_segment is None or len(vel_segment) < 2:
            return 'full_body'

        mean_vel = vel_segment.mean(axis=0)   # shape (13,)

        hip_vel   = mean_vel[[6,7]].mean()
        knee_vel  = mean_vel[[8,9]].mean()
        ankle_vel = mean_vel[[10,11]].mean()
        arm_vel   = mean_vel[[0,1,2,3,4,5]].mean()

        # Bouncing — vertical knee/hip dominant, ankles stable
        if knee_vel > ankle_vel * 1.3 and knee_vel > arm_vel * 1.2:
            return 'bouncing'

        # Stepping — ankles dominant, fast foot movement
        if ankle_vel > knee_vel * 1.2 and ankle_vel > arm_vel * 1.5:
            return 'stepping'

        # Sliding — hips + ankles together, lower velocity overall
        if hip_vel > arm_vel * 1.2 and ankle_vel > arm_vel * 1.0:
            return 'sliding'

        return 'full_body'

    def classify_sequence(self, velocity: np.ndarray, window_sec: float = 1.0,
                           fps: float = 30.0) -> list:
        """
        Classify the full velocity sequence in sliding windows.

        Returns a list of (start_frame, end_frame, move_type) tuples.
        """
        win = max(2, int(window_sec * fps))
        n = len(velocity)
        segments = []
        i = 0
        while i < n:
            seg = velocity[i: i + win]
            move = self.classify_segment(seg)
            segments.append((i, min(i + win, n), move))
            i += win
        return segments

    def get_mask_for_move(self, move_type: str) -> np.ndarray:
        """Return normalised joint weight mask for the given move type."""
        return self._NORM_MASKS.get(move_type, self._NORM_MASKS['full_body'])

    def get_profile(self, move_type: str) -> dict:
        """Return the scoring profile dict for the given move type."""
        return self.MOVE_PROFILES.get(move_type, self.MOVE_PROFILES['full_body'])

    def get_frame_masks(self, velocity: np.ndarray, fps: float = 30.0) -> np.ndarray:
        """
        Return per-frame mask array (shape N×13).
        Each row is the normalised joint weight mask for that frame's move type.
        """
        segments = self.classify_sequence(velocity, fps=fps)
        masks = np.ones((len(velocity), 13), dtype=float)
        for start, end, move in segments:
            masks[start:end] = self.get_mask_for_move(move)
        return masks


# =====================================================================
# WIRAGA TIME-SERIES ANALYSIS HELPERS
# Implements expert framework: turning points (bouncing), foot strikes
# (stepping), linear stability (sliding).
# =====================================================================

def _extract_landmark_axis(landmarks, landmark_id, axis='y'):
    """Extract x or y for one landmark across all frames.

    Format: landmarks[frame][landmark_id] = {'x': float, 'y': float, 'v': float}
    Returns np.array shape (N,), or empty array on any failure.
    """
    if landmarks is None or len(landmarks) == 0:
        return np.array([])
    if axis not in ('x', 'y'):
        return np.array([])
    values = []
    for frame in landmarks:
        if not isinstance(frame, list) or len(frame) <= landmark_id:
            continue
        lm = frame[landmark_id]
        if not isinstance(lm, dict) or axis not in lm:
            continue
        values.append(lm[axis])
    return np.array(values, dtype=float)


def _detect_bouncing_turning_points(knee_angles, hip_y_coords, fps=30.0):
    """Detect bottom peaks of bouncing motion (deepest knee bends).

    Returns list of {'frame', 'time', 'knee_depth', 'hip_y'}.
    """
    if knee_angles is None or len(knee_angles) < 10:
        return []
    knee_signal = knee_angles.mean(axis=1) if knee_angles.ndim == 2 else knee_angles
    inv_knee = -knee_signal
    prom = max(2.0, float(np.std(knee_signal)) * 0.3)
    min_dist = max(1, int(fps * 0.25))
    peaks, _ = find_peaks(inv_knee, prominence=prom, distance=min_dist)
    turning_points = []
    for fi in peaks:
        if fi >= len(knee_signal):
            continue
        turning_points.append({
            'frame':      int(fi),
            'time':       float(fi / fps),
            'knee_depth': float(knee_signal[fi]),
            'hip_y':      float(hip_y_coords[fi])
                          if hip_y_coords is not None and fi < len(hip_y_coords)
                          else None,
        })
    return turning_points


def _detect_foot_strikes(ankle_y_coords, fps=30.0, stationary_thresh=0.003,
                         min_stationary_frames=3):
    """Detect foot-strike moments — ankle Y stops descending (lands on floor).

    Returns list of {'frame', 'time'}.
    """
    if ankle_y_coords is None or len(ankle_y_coords) < min_stationary_frames + 2:
        return []
    ay = np.asarray(ankle_y_coords)
    dy = np.diff(ay)
    strikes = []
    i = 1
    while i < len(dy) - min_stationary_frames:
        descending = dy[i - 1] > 0.0015
        static_window = dy[i: i + min_stationary_frames]
        all_static = np.all(np.abs(static_window) < stationary_thresh)
        if descending and all_static:
            strikes.append({'frame': int(i), 'time': float(i / fps)})
            i += min_stationary_frames
        else:
            i += 1
    return strikes


def _detect_weight_shift(hip_x_coords, ankle_x_coords, fps=30.0):
    """Pearson correlation between hip X and ankle X lateral motion.

    Returns float [0, 1]: 1.0 = perfect weight shift, 0.0 = hip stays static.
    """
    if hip_x_coords is None or ankle_x_coords is None:
        return 0.0
    if len(hip_x_coords) < 5 or len(ankle_x_coords) < 5:
        return 0.0
    h = np.asarray(hip_x_coords) - np.mean(hip_x_coords)
    a = np.asarray(ankle_x_coords) - np.mean(ankle_x_coords)
    h_std, a_std = h.std(), a.std()
    if h_std < 1e-6 or a_std < 1e-6:
        return 0.0
    return max(0.0, float(np.mean(h * a) / (h_std * a_std)))


def _measure_sliding_linearity(ankle_distance_x):
    """R² of linear fit on ankle distance over time.

    Returns float [0, 1]: 1.0 = perfectly linear slide.
    """
    if ankle_distance_x is None or len(ankle_distance_x) < 5:
        return 0.0
    d = np.asarray(ankle_distance_x)
    x = np.arange(len(d), dtype=float)
    slope, intercept = np.polyfit(x, d, 1)
    d_pred = slope * x + intercept
    ss_res = np.sum((d - d_pred) ** 2)
    ss_tot = np.sum((d - d.mean()) ** 2)
    if ss_tot < 1e-9:
        return 0.0
    return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))


def _measure_hip_y_stability(hip_y_coords):
    """Stability of hip Y over time — 1.0 = flat (no bob), 0.0 = bouncing.

    Maps coefficient of variation: CV=0 → 1.0, CV≥0.1 → 0.0.
    """
    if hip_y_coords is None or len(hip_y_coords) < 5:
        return 0.0
    hy = np.asarray(hip_y_coords)
    mean = abs(hy.mean()) + 1e-6
    cv = hy.std() / mean
    return max(0.0, 1.0 - cv * 10.0)


class HipHopDanceScorer:
    """
    Expert-aligned Hip Hop dance scoring engine.

    Three pillars (matching professional judging criteria):
      1. Timing / Musicality  (35%) — do movement peaks land on the beat?
      2. Movement Quality     (35%) — is the motion pattern similar to teacher?
      3. Power / Dynamics     (30%) — are hits sharp and contrasting?

    Angle comparison is used only as a *shape similarity* signal within
    Movement Quality, NOT as the primary score.  This avoids penalising
    students for proportional body differences.

    Joint scoring is context-aware: a MoveClassifier detects what move is
    happening each second and applies a mask so only relevant joints count.
    E.g. during a bounce, wrists score near-zero; during an arm wave,
    knees score near-zero.
    """

    # Fallback weight vector used when no move mask is active (13 joints)
    JOINT_WEIGHTS = np.array([
        0.7, 0.7,   # 0-1  Elbows
        1.0, 1.0,   # 2-3  Shoulders
        0.3, 0.3,   # 4-5  Wrists
        1.2, 1.2,   # 6-7  Hips
        1.1, 1.1,   # 8-9  Knees
        0.3, 0.3,   # 10-11 Ankles
        1.5,        # 12   Spine
    ])
    JOINT_WEIGHTS = JOINT_WEIGHTS / JOINT_WEIGHTS.sum()

    # Max ROM for normalisation (degrees)
    MAX_ROM = np.array([150,150, 180,180, 120,120, 160,160, 160,160, 90,90, 90])

    # Region → column indices
    REGIONS = {
        'arms':  [0, 1, 2, 3, 4, 5],
        'legs':  [6, 7, 8, 9, 10, 11],
        'torso': [12],
    }

    def __init__(self):
        self.classifier = MoveClassifier()

    # ── 1. TIMING / MUSICALITY ────────────────────────────────────────────────

    def score_timing(self,
                     student_vel: np.ndarray, student_ts: np.ndarray,
                     teacher_vel: np.ndarray, teacher_ts: np.ndarray,
                     student_beats: np.ndarray, teacher_beats: np.ndarray,
                     window_sec: float = 0.18) -> dict:
        """
        Measure how well movement peaks align with musical beats.

        For each beat in the teacher recording we find the nearest teacher
        velocity spike; we then check whether the student has a spike within
        ±WINDOW_SEC of the corresponding student beat.  Score = hit-rate × 100.

        Falls back to cross-correlation of velocity envelopes when beat
        detection yields too few events (e.g. no audio).
        """
        WINDOW_SEC = window_sec   # Dynamic tolerance based on movement config

        # Move-aware scalar velocity envelope.
        # Per-frame masks ensure e.g. wrists score near-zero during a bounce.
        def _masked_env(vel):
            masks = self.classifier.get_frame_masks(vel)
            n = min(len(vel), len(masks))
            return (vel[:n] * masks[:n]).sum(axis=1)

        s_env = _masked_env(student_vel)
        t_env = _masked_env(teacher_vel)

        # --- Beat-alignment path ---
        if len(student_beats) >= 4 and len(teacher_beats) >= 4:
            hits = 0
            total = 0
            for tb in teacher_beats:
                # nearest teacher velocity peak around this beat
                t_mask = np.abs(teacher_ts[:len(t_env)] - tb) < WINDOW_SEC
                if not t_mask.any():
                    continue
                # check if student has a matching spike
                sb = student_beats[np.argmin(np.abs(student_beats - tb))]
                s_mask = np.abs(student_ts[:len(s_env)] - sb) < WINDOW_SEC
                t_peak = t_env[t_mask].max()
                s_peak = s_env[s_mask].max() if s_mask.any() else 0.0
                total += 1
                # student must reach ≥ 50% of teacher's peak at that beat
                if s_mask.any() and s_peak >= 0.5 * t_peak:
                    hits += 1

            if total > 0:
                raw_score = hits / total
                timing_score = max(0.0, min(100.0, raw_score * 100))
                return {'timing_score': round(timing_score, 2),
                        'beats_matched': hits, 'beats_total': total,
                        'method': 'beat_alignment'}

        # --- Fallback: cross-correlation of velocity envelopes ---
        min_len = min(len(s_env), len(t_env), 3000)  # cap for speed
        s_clip = s_env[:min_len]
        t_clip = t_env[:min_len]

        def _norm(x):
            x = x - x.mean()
            std = x.std()
            return x / std if std > 1e-9 else x

        corr = np.correlate(_norm(s_clip), _norm(t_clip), mode='full')
        # Best-lag similarity [-1, 1] → [0, 100]
        best_sim = float(corr.max()) / max(min_len, 1)
        timing_score = max(0.0, min(100.0, (best_sim + 1) * 50))
        return {'timing_score': round(timing_score, 2),
                'method': 'envelope_correlation'}

    # ── 2. MOVEMENT QUALITY ───────────────────────────────────────────────────

    def score_movement_quality(self,
                                student_data: np.ndarray, student_ts: np.ndarray,
                                teacher_data: np.ndarray, teacher_ts: np.ndarray,
                                student_vel: np.ndarray, teacher_vel: np.ndarray,
                                core_joints: list = None) -> dict:
        """
        Measure shape similarity of the motion pattern — NOT raw angle values.

        Two sub-components, averaged:
          A. Velocity-pattern DTW: compare the velocity time-series (shape of
             the movement) rather than absolute angles.  This is body-proportion
             agnostic.
          B. Normalised angle DTW: compare pose shapes with ROM-normalisation.
             This penalises consistently wrong poses (e.g. never bending knees)
             but is less sensitive to proportional differences.

        DTW window is constrained to 10% of sequence length to prevent
        excessive time-stretching that masks timing errors.
        
        PERBAIKAN 3: FastDTW Optimization — Videos > 20 sec are downsampled to 12 fps
        to reduce computational load while preserving motion patterns.
        """
        student_data = np.array(student_data, dtype=float)
        teacher_data = np.array(teacher_data, dtype=float)

        if len(student_data) < 2 or len(teacher_data) < 2:
            return {'movement_score': 0, 'regional': {r: 0 for r in self.REGIONS},
                    'velocity_score': 0, 'pose_shape_score': 0}

        # ── PERBAIKAN 3: Downsampling untuk video yang terlalu panjang ──
        # Jika durasi video > 20 detik, downsample ke 12 fps untuk menghemat compute
        s_duration = student_ts[-1] if len(student_ts) > 0 else 0
        t_duration = teacher_ts[-1] if len(teacher_ts) > 0 else 0
        max_duration = max(s_duration, t_duration)
        
        if max_duration > 20.0:  # longer than 20 seconds
            print(f'[score_movement] Video duration {max_duration:.1f}s exceeds 20s threshold — downsampling to 12 fps')
            student_data_dtw, student_ts_dtw = downsample_pose_data(student_data, student_ts, target_fps=12.0)
            teacher_data_dtw, teacher_ts_dtw = downsample_pose_data(teacher_data, teacher_ts, target_fps=12.0)
            student_vel_dtw = student_vel[::max(1, len(student_vel) // len(student_data_dtw))] if len(student_vel) > len(student_data_dtw) else student_vel
            teacher_vel_dtw = teacher_vel[::max(1, len(teacher_vel) // len(teacher_data_dtw))] if len(teacher_vel) > len(teacher_data_dtw) else teacher_vel
        else:
            student_data_dtw, student_ts_dtw = student_data, student_ts
            teacher_data_dtw, teacher_ts_dtw = teacher_data, teacher_ts
            student_vel_dtw, teacher_vel_dtw = student_vel, teacher_vel

        explicit_joint_mask = None
        if core_joints is not None:
            explicit_joint_mask = np.zeros(13, dtype=float)
            valid_core_joints = [idx for idx in core_joints if 0 <= idx < 13]
            if valid_core_joints:
                explicit_joint_mask[valid_core_joints] = self.JOINT_WEIGHTS[valid_core_joints]
                explicit_joint_mask = explicit_joint_mask / explicit_joint_mask.sum()

        # ── Build per-frame move masks from TEACHER velocity (ground truth move) ──
        # We use the teacher's movement to define what type of move is happening,
        # then apply that mask to both teacher and student so irrelevant joints
        # (e.g. wrists during a bounce) don't affect the score.
        t_masks = self.classifier.get_frame_masks(teacher_vel_dtw)  # shape (T, 13)
        # Pad or trim to match teacher_data length
        n_t = len(teacher_data_dtw)
        if len(t_masks) < n_t:
            t_masks = np.vstack([t_masks, np.tile(t_masks[-1], (n_t - len(t_masks), 1))])
        else:
            t_masks = t_masks[:n_t]

        # ── A. Velocity pattern DTW (move-context-aware) ──
        # Frame index tracker so we can look up the teacher mask at DTW warp position j
        def _vel_dist(a, b):
            # a = student frame velocity (13,), b = teacher frame velocity (13,)
            # If movement_type was selected by user, score only configured core joints.
            joint_weights = explicit_joint_mask if explicit_joint_mask is not None else self.JOINT_WEIGHTS
            diff = (a - b) * joint_weights
            return float(np.sqrt((diff ** 2).sum()))

        try:
            v_dist, v_path = fastdtw(student_vel_dtw, teacher_vel_dtw, dist=_vel_dist)
            # Re-weight path errors by teacher move mask
            masked_v_errors = []
            for i, j in v_path:
                si = min(i, len(student_vel_dtw) - 1)
                tj = min(j, len(teacher_vel_dtw) - 1)
                diff = np.abs(student_vel_dtw[si] - teacher_vel_dtw[tj])
                if explicit_joint_mask is not None:
                    mask = explicit_joint_mask
                else:
                    mask = self.classifier.get_mask_for_move(
                        self.classifier.classify_segment(teacher_vel_dtw[max(0,tj-5):tj+5])
                    )
                masked_v_errors.append(float((diff * mask).sum()))
            v_avg = float(np.mean(masked_v_errors)) if masked_v_errors else 1.0
            velocity_score = max(0.0, min(100.0, (1 - v_avg / 80.0) * 100))
        except Exception as e:
            print(f'[score_movement] Velocity DTW error: {e}')
            velocity_score = 50.0
            v_path = []

        # ── B. Normalised angle DTW (move-context-aware) ──
        def _angle_dist(a, b):
            norm_a = a / self.MAX_ROM
            norm_b = b / self.MAX_ROM
            joint_weights = explicit_joint_mask if explicit_joint_mask is not None else self.JOINT_WEIGHTS
            diff = (norm_a - norm_b) * joint_weights
            return float(np.sqrt((diff ** 2).sum()))

        try:
            a_dist, a_path = fastdtw(student_data_dtw, teacher_data_dtw, dist=_angle_dist)
            a_avg = a_dist / max(len(a_path), 1)
            pose_shape_score = max(0.0, min(100.0, (1 - a_avg) * 100))
        except Exception as e:
            print(f'[score_movement] Angle DTW error: {e}')
            pose_shape_score = 50.0
            a_path = []

        movement_score = 0.55 * velocity_score + 0.45 * pose_shape_score

        # ── Regional breakdown using angle path + move masks ──
        # Each frame's error is weighted by the teacher's move mask so regions
        # that are irrelevant for that move type contribute less to the score.
        regional = {}
        if a_path:
            s_al  = np.array([student_data_dtw[i]  for i, _ in a_path], dtype=float)
            t_al  = np.array([teacher_data_dtw[j]  for _, j in a_path], dtype=float)
            # Per-path-step move mask from teacher velocity
            if explicit_joint_mask is not None:
                path_masks = np.tile(explicit_joint_mask, (len(a_path), 1))
            else:
                path_masks = np.array([
                    self.classifier.get_mask_for_move(
                        self.classifier.classify_segment(
                            teacher_vel_dtw[max(0, j-5): j+5] if j < len(teacher_vel_dtw) else teacher_vel_dtw[-5:]
                        )
                    )
                    for _, j in a_path
                ])  # shape (P, 13)

            norm_err = np.abs(s_al - t_al) / self.MAX_ROM  # (P, 13)
            weighted_err = norm_err * path_masks            # (P, 13) — irrelevant joints near-zero
            mean_err = weighted_err.mean(axis=0)            # (13,)

            for region, idx in self.REGIONS.items():
                # Sum of mask weights for this region (to normalise fairly)
                region_mask_sum = path_masks[:, idx].mean()
                if region_mask_sum < 0.01:
                    # Region was entirely masked out for this routine — give benefit of doubt
                    regional[region] = 100.0
                else:
                    regional[region] = round(
                        max(0.0, min(100.0, (1 - np.mean(mean_err[idx]) / (region_mask_sum + 1e-9)) * 100)), 2
                    )
        else:
            regional = {r: round(movement_score, 2) for r in self.REGIONS}

        return {
            'movement_score': round(movement_score, 2),
            'velocity_score': round(velocity_score, 2),
            'pose_shape_score': round(pose_shape_score, 2),
            'regional': regional,
        }

    # ── 3. POWER / DYNAMICS ───────────────────────────────────────────────────

    def score_power(self,
                    student_acc: np.ndarray, teacher_acc: np.ndarray,
                    student_vel: np.ndarray, teacher_vel: np.ndarray) -> dict:
        """
        Measure sharpness and contrast of movement hits.

        Two sub-scores:
          A. Hit sharpness — ratio of student 90th-percentile acceleration
             peaks to teacher peaks (weighted by joint importance).
             Value < 1 means student's hits are softer than teacher's.

          B. Dynamic contrast — std-dev of velocity envelope.
             High std = dancer alternates between stillness and explosive moves.
             Low std = monotone energy throughout.

        Both are ratio-based so they are body-proportion agnostic.
        """
        def _weighted_scalar(arr):
            return (np.clip(arr, 0, None) * self.JOINT_WEIGHTS).sum(axis=1)

        s_acc_scalar = _weighted_scalar(student_acc)
        t_acc_scalar = _weighted_scalar(teacher_acc)
        s_vel_scalar = _weighted_scalar(student_vel)
        t_vel_scalar = _weighted_scalar(teacher_vel)

        # ── A. Hit sharpness ──
        t_p90 = max(np.percentile(t_acc_scalar, 90), 1e-6)
        s_p90 = np.percentile(s_acc_scalar, 90)
        sharpness_ratio = min(s_p90 / t_p90, 1.3)   # cap at 130% — reward up to 30% overshoot
        sharpness_score = max(0.0, min(100.0, sharpness_ratio * 100))

        # ── B. Dynamic contrast ──
        t_contrast = max(t_vel_scalar.std(), 1e-6)
        s_contrast = s_vel_scalar.std()
        contrast_ratio = min(s_contrast / t_contrast, 1.3)
        contrast_score = max(0.0, min(100.0, contrast_ratio * 100))

        power_score = 0.6 * sharpness_score + 0.4 * contrast_score

        return {
            'power_score': round(power_score, 2),
            'sharpness_score': round(sharpness_score, 2),
            'contrast_score': round(contrast_score, 2),
        }

    # ── MAIN EVALUATE ─────────────────────────────────────────────────────────

    def evaluate(self,
                 student_data: np.ndarray, student_ts: np.ndarray,
                 teacher_data: np.ndarray, teacher_ts: np.ndarray,
                 student_landmarks: list = None,
                 teacher_landmarks: list = None,
                 student_video_path: str = None,
                 teacher_video_path: str = None,
                 selected_movement: str = None) -> dict:
        """
        Full evaluation pipeline.

        Returns a dict with:
          overall_score  — weighted composite (0-100)
          timing_score   — musicality component
          movement_score — shape/velocity component
          power_score    — sharpness/dynamics component
          regional       — {arms, legs, torso} sub-scores
          detail         — per-component detail for UI display
        """
        if selected_movement not in MOVEMENT_CONFIG:
            raise ValueError(
                f"Invalid movement_type '{selected_movement}'. "
                f"Valid movement_type values: {', '.join(MOVEMENT_CONFIG.keys())}"
            )

        config = MOVEMENT_CONFIG[selected_movement]
        core_joints = config['core_joints']

        student_data = np.array(student_data, dtype=float)
        teacher_data = np.array(teacher_data, dtype=float)

        empty = {
            'overall_score': 0, 'timing_score': 0,
            'movement_score': 0, 'power_score': 0,
            'regional': {'arms': 0, 'legs': 0, 'torso': 0},
            'detail': {},
        }

        if len(student_data) < 3 or len(teacher_data) < 3:
            return empty

        # ── Kinematics ──
        s_vel, s_acc = compute_joint_kinematics(student_data, student_ts)
        t_vel, t_acc = compute_joint_kinematics(teacher_data, teacher_ts)
        if s_vel is None or t_vel is None:
            return empty

        # ── Beat extraction (best-effort) ──
        s_beats = extract_beat_times(student_video_path) if student_video_path else np.array([])
        t_beats = extract_beat_times(teacher_video_path) if teacher_video_path else np.array([])

        # ── Move classification (for UI display + context-aware feedback) ──
        fps_est = len(student_data) / float(student_ts[-1]) if student_ts[-1] > 0 else 30.0
        move_segments = self.classifier.classify_sequence(s_vel, fps=fps_est)
        # Convert to serialisable list of {start_sec, end_sec, move}
        move_timeline = [
            {
                'start_sec': round(student_ts[min(s, len(student_ts)-1)], 2),
                'end_sec':   round(student_ts[min(e-1, len(student_ts)-1)], 2),
                'move':      m,
            }
            for s, e, m in move_segments
        ]

        # ── Three-pillar scoring ──
        # Keep automatic classification for UI timeline only. The scoring rubric
        # uses the movement_type selected by the user.
        from collections import Counter
        move_counts = Counter([m for s, e, m in move_segments])
        timing_tolerance = config['thresholds'].get('timing_tolerance', 0.18)

        t_res = self.score_timing(
            s_vel, student_ts[:len(s_vel)],
            t_vel, teacher_ts[:len(t_vel)],
            s_beats, t_beats,
            window_sec=timing_tolerance
        )
        # For bouncing, symmetrize bilateral joint pairs before DTW so left/right
        # asymmetry from camera angle doesn't penalize a correct bilateral bounce.
        if selected_movement == 'bouncing':
            def _symmetrize(arr):
                a = arr.copy()
                for l, r in [(6,7),(8,9),(10,11)]:  # hips, knees, ankles
                    avg = (a[:, l] + a[:, r]) / 2
                    a[:, l] = avg; a[:, r] = avg
                return a
            sd_dtw = _symmetrize(student_data)
            td_dtw = _symmetrize(teacher_data)
            sv_dtw = _symmetrize(s_vel)
            tv_dtw = _symmetrize(t_vel)
        else:
            sd_dtw, td_dtw, sv_dtw, tv_dtw = student_data, teacher_data, s_vel, t_vel

        m_res = self.score_movement_quality(
            sd_dtw, student_ts,
            td_dtw, teacher_ts,
            sv_dtw, tv_dtw,
            core_joints=core_joints,
        )
        p_res = self.score_power(s_acc, t_acc, s_vel, t_vel)

        timing_score   = t_res['timing_score']
        power_score    = p_res['power_score']
        
        # ── WIRAGA SCORING — time-series analysis per movement ──────────────
        # Expert framework: turning points (bouncing), foot strikes (stepping),
        # linear stability (sliding).  Falls back to DTW if landmarks absent.
        fps = 30.0
        wiraga_breakdown = {'movement_type': selected_movement, 'components': {}}

        if selected_movement == 'bouncing':
            target = config['thresholds']['knee_bend_target']
            student_knees = student_data[:, [8, 9]]
            student_hip_y = None
            if student_landmarks:
                l_hip_y = _extract_landmark_axis(student_landmarks, 23, 'y')
                r_hip_y = _extract_landmark_axis(student_landmarks, 24, 'y')
                if len(l_hip_y) > 0 and len(r_hip_y) > 0:
                    mn = min(len(l_hip_y), len(r_hip_y))
                    student_hip_y = (l_hip_y[:mn] + r_hip_y[:mn]) / 2.0

            turning_points = _detect_bouncing_turning_points(
                student_knees, student_hip_y, fps=fps
            )
            if not turning_points:
                wiraga_score = 0.0
                successful_bends = 0
            else:
                successful_bends = sum(
                    1 for tp in turning_points if tp['knee_depth'] <= target
                )
                wiraga_score = float(successful_bends / len(turning_points)) * 100.0
            print(f'[bouncing] {len(turning_points)} turning points, '
                  f'{successful_bends} reached target (≤{target}°)')
            wiraga_breakdown['components'] = {
                'turning_points_count': len(turning_points),
                'successful_bends': successful_bends,
            }

        elif selected_movement == 'stepping':
            if not student_landmarks:
                wiraga_score = m_res['movement_score']
                wiraga_breakdown['components'] = {'fallback': 'no_landmarks'}
            else:
                l_ankle_y = _extract_landmark_axis(student_landmarks, 27, 'y')
                r_ankle_y = _extract_landmark_axis(student_landmarks, 28, 'y')
                strikes_l = _detect_foot_strikes(l_ankle_y, fps=fps)
                strikes_r = _detect_foot_strikes(r_ankle_y, fps=fps)
                total_strikes = len(strikes_l) + len(strikes_r)
                if len(l_ankle_y) > 0 and len(r_ankle_y) > 0:
                    print(f'[stepping debug] L_ankle_y range: {l_ankle_y.min():.3f}–{l_ankle_y.max():.3f}, '
                          f'R_ankle_y range: {r_ankle_y.min():.3f}–{r_ankle_y.max():.3f}')

                teacher_strikes = 0
                if teacher_landmarks:
                    t_l_y = _extract_landmark_axis(teacher_landmarks, 27, 'y')
                    t_r_y = _extract_landmark_axis(teacher_landmarks, 28, 'y')
                    teacher_strikes = (len(_detect_foot_strikes(t_l_y, fps=fps)) +
                                       len(_detect_foot_strikes(t_r_y, fps=fps)))

                if teacher_strikes > 0:
                    strike_ratio = min(total_strikes / teacher_strikes, 1.0)
                else:
                    strike_ratio = min(total_strikes / 4.0, 1.0)
                strike_accuracy = strike_ratio * 100.0

                l_hip_x = _extract_landmark_axis(student_landmarks, 23, 'x')
                r_hip_x = _extract_landmark_axis(student_landmarks, 24, 'x')
                weight_shift_score = 50.0
                if len(l_hip_x) > 0 and len(r_hip_x) > 0:
                    mn = min(len(l_hip_x), len(r_hip_x))
                    mid_hip_x = (l_hip_x[:mn] + r_hip_x[:mn]) / 2.0
                    l_ankle_x = _extract_landmark_axis(student_landmarks, 27, 'x')
                    r_ankle_x = _extract_landmark_axis(student_landmarks, 28, 'x')
                    if len(l_ankle_x) > 0 and len(r_ankle_x) > 0:
                        primary_ankle = (l_ankle_x if l_ankle_x.std() > r_ankle_x.std()
                                         else r_ankle_x)
                    elif len(l_ankle_x) > 0:
                        primary_ankle = l_ankle_x
                    elif len(r_ankle_x) > 0:
                        primary_ankle = r_ankle_x
                    else:
                        primary_ankle = np.array([])
                    if len(primary_ankle) >= 5:
                        weight_shift_score = _detect_weight_shift(
                            mid_hip_x, primary_ankle, fps=fps
                        ) * 100.0

                wiraga_score = strike_accuracy * 0.5 + weight_shift_score * 0.5
                print(f'[stepping] strikes={total_strikes} (teacher={teacher_strikes}), '
                      f'strike_acc={strike_accuracy:.1f}, weight_shift={weight_shift_score:.1f}')
                wiraga_breakdown['components'] = {
                    'foot_strikes':       total_strikes,
                    'teacher_strikes':    teacher_strikes,
                    'strike_accuracy':    round(strike_accuracy, 1),
                    'weight_shift_score': round(weight_shift_score, 1),
                }

        elif selected_movement == 'sliding':
            if not student_landmarks:
                wiraga_score = m_res['movement_score']
                wiraga_breakdown['components'] = {'fallback': 'no_landmarks'}
            else:
                l_ankle_x = _extract_landmark_axis(student_landmarks, 27, 'x')
                r_ankle_x = _extract_landmark_axis(student_landmarks, 28, 'x')
                l_hip_y   = _extract_landmark_axis(student_landmarks, 23, 'y')
                r_hip_y   = _extract_landmark_axis(student_landmarks, 24, 'y')
                l_sh_x    = _extract_landmark_axis(student_landmarks, 11, 'x')
                r_sh_x    = _extract_landmark_axis(student_landmarks, 12, 'x')

                if (len(l_ankle_x) > 0 and len(r_ankle_x) > 0 and
                        len(l_hip_y) > 0 and len(r_hip_y) > 0):
                    mn = min(len(l_ankle_x), len(r_ankle_x),
                             len(l_hip_y), len(r_hip_y))
                    ankle_dist = np.abs(l_ankle_x[:mn] - r_ankle_x[:mn])
                    mid_hip_y  = (l_hip_y[:mn] + r_hip_y[:mn]) / 2.0

                    linearity_score  = _measure_sliding_linearity(ankle_dist) * 100.0
                    stability_score  = _measure_hip_y_stability(mid_hip_y) * 100.0

                    reach_score = 50.0
                    if len(l_sh_x) > 0 and len(r_sh_x) > 0:
                        sh_width = np.abs(
                            l_sh_x[:mn] - r_sh_x[:mn]
                        ).mean()
                        if sh_width > 1e-6:
                            target_ratio = config['thresholds']['sliding_dist_shoulder_ratio']
                            reach_score = min(ankle_dist.max() / sh_width / target_ratio, 1.0) * 100.0

                    wiraga_score = (linearity_score + stability_score + reach_score) / 3.0
                    print(f'[sliding] linearity={linearity_score:.1f}, '
                          f'stability={stability_score:.1f}, reach={reach_score:.1f}')
                    wiraga_breakdown['components'] = {
                        'linearity':    round(linearity_score, 1),
                        'hip_stability': round(stability_score, 1),
                        'reach_ratio':  round(reach_score, 1),
                    }
                else:
                    wiraga_score = m_res['movement_score']
                    wiraga_breakdown['components'] = {'fallback': 'insufficient_landmarks'}

        else:
            wiraga_score = m_res['movement_score']
            wiraga_breakdown['components'] = {'fallback': 'unknown_movement'}

        wirama_score = timing_score
        movement_score = wiraga_score

        # Weighted composite menggunakan bobot dari MOVEMENT_CONFIG
        # Ini adalah implementasi EXPERT JUDGMENT MATRIX yang memenuhi skripsi requirements
        weight_wiraga = config['weights']['wiraga']
        weight_wirama = config['weights']['wirama']
        weight_power = config['weights']['power']

        # For Coach basic moves (Wiraga 50% / Wirama 50%), Power dimension is
        # not part of the expert validation framework. Skip its contribution.
        if weight_power <= 0.0:
            power_score = 0.0

        overall_score = (
            (weight_wiraga * wiraga_score) +
            (weight_wirama * wirama_score) +
            (weight_power * power_score)
        )
        overall_score = round(max(0.0, min(100.0, overall_score)), 2)

        return {
            'overall_score': overall_score,
            'timing_score':  round(timing_score,   2),
            'movement_score': round(movement_score, 2),
            'power_score':   round(power_score,     2),
            'power_weight':  weight_power,
            # regional breakdown comes from movement quality sub-scores
            'regional': m_res['regional'],
            'detail': {
                'timing':   t_res,
                'movement': m_res,
                'power':    p_res,
                'movement_config': {
                    'selected_movement': selected_movement,
                    'core_joints': core_joints,
                    'weights': config['weights'],
                },
            },
            'move_timeline': move_timeline,
            'selected_movement': selected_movement,
            '_s_vel':          s_vel,
            '_s_acc':          s_acc,
            '_t_vel':          t_vel,
            '_t_acc':          t_acc,
            '_s_beats':        s_beats,
            'wiraga_breakdown': wiraga_breakdown,
        }


def _wiraga_message(joint_name, diff, movement):
    """Natural language message based on joint and direction."""
    magnitude = abs(diff)
    degree_str = f' (off by ~{int(magnitude)}°)' if magnitude > 15 else ''

    if 'knee' in joint_name:
        if diff > 0:
            return f'Bend your {joint_name} deeper{degree_str}.'
        else:
            return f'Don\'t over-bend your {joint_name}{degree_str}.'

    if 'hip' in joint_name:
        if diff > 0:
            return f'Lower your {joint_name} position — engage your core more{degree_str}.'
        else:
            return f'Stand taller — your {joint_name} is too low{degree_str}.'

    if 'shoulder' in joint_name:
        if diff > 0:
            return f'Relax your {joint_name} — it\'s too tense or raised{degree_str}.'
        else:
            return f'Lift your {joint_name} slightly{degree_str}.'

    if 'ankle' in joint_name:
        if diff > 0:
            return f'Flatten your {joint_name} — keep feet grounded{degree_str}.'
        else:
            return f'Lift on your toes more with your {joint_name}{degree_str}.'

    if 'spine' in joint_name:
        if diff > 0:
            return f'Stand up straighter — your posture is leaning{degree_str}.'
        else:
            return f'Lean forward slightly to engage the movement{degree_str}.'

    return f'Adjust your {joint_name} position{degree_str}.'


def _find_missed_beats(student_vel, student_ts, student_beats, move_mask,
                       window_sec=0.18):
    """Return (miss_rate, occurrences) for music beats the student failed to hit.

    For each musical beat extracted from the student's audio, checks whether
    the student's masked velocity envelope has a movement peak within
    ±window_sec.  Beats with no matching peak are flagged as missed.

    Returns:
        miss_rate   — float [0, 1], fraction of beats with no student peak
        occurrences — list of {start_time, end_time, severity}
    """
    from scipy.signal import find_peaks

    if len(student_beats) < 4:
        return 0.0, []

    s_env = (student_vel * move_mask).sum(axis=1)
    if len(s_env) < 5:
        return 0.0, []

    s_std = float(np.std(s_env))
    if s_std == 0:
        return 0.0, []

    # Student movement impulses: peaks in the masked velocity envelope
    s_peaks, _ = find_peaks(s_env, prominence=s_std * 0.3,
                            distance=int(0.1 * 30))
    s_peak_times = student_ts[s_peaks] if len(s_peaks) else np.array([])

    missed = 0
    occurrences = []
    for beat_t in student_beats:
        beat_t = float(beat_t)
        if len(s_peak_times) == 0:
            missed += 1
            occurrences.append({
                'start_time': round(beat_t - 0.1, 2),
                'end_time':   round(beat_t + 0.1, 2),
                'severity':   70.0,
            })
            continue
        nearest = float(s_peak_times[np.argmin(np.abs(s_peak_times - beat_t))])
        gap = abs(nearest - beat_t)
        if gap > window_sec:
            missed += 1
            occurrences.append({
                'start_time': round(beat_t - 0.1, 2),
                'end_time':   round(beat_t + 0.1, 2),
                'severity':   round(min(100.0, gap / window_sec * 50), 1),
            })

    miss_rate = missed / len(student_beats)
    return miss_rate, occurrences


def _find_joint_occurrences(student_data, teacher_data, joint_idx,
                             student_ts, teacher_ts, threshold_deg=15,
                             min_duration_sec=0.3):
    """Find time ranges where joint angle differs significantly (DTW-aligned)."""
    try:
        _, path = fastdtw(student_data, teacher_data, dist=euclidean)
    except Exception:
        return []

    occurrences = []
    run = []
    for s_idx, t_idx in sorted(path, key=lambda p: p[0]):
        if s_idx >= len(student_ts) or t_idx >= len(teacher_data):
            continue
        diff = student_data[s_idx, joint_idx] - teacher_data[t_idx, joint_idx]
        t_time = float(student_ts[s_idx])
        if abs(diff) >= threshold_deg:
            run.append((t_time, abs(diff)))
        else:
            if run:
                start, end = run[0][0], run[-1][0]
                if (end - start) >= min_duration_sec:
                    avg_sev = sum(d for _, d in run) / len(run)
                    occurrences.append({
                        'start_time': round(start, 2),
                        'end_time':   round(end, 2),
                        'severity':   round(min(100, avg_sev * 2), 1),
                    })
                run = []
    if run:
        start, end = run[0][0], run[-1][0]
        if (end - start) >= min_duration_sec:
            avg_sev = sum(d for _, d in run) / len(run)
            occurrences.append({
                'start_time': round(start, 2),
                'end_time':   round(end, 2),
                'severity':   round(min(100, avg_sev * 2), 1),
            })
    return occurrences




def generate_generalized_feedback(student_data, student_ts, teacher_data,
                                   teacher_ts, student_vel, teacher_vel,
                                   selected_movement, student_beats=None,
                                   wiraga_breakdown=None):
    """Aggregate feedback across all repetitions into generalized issues.

    Returns list of dicts sorted by severity (worst first):
      [{'type': 'wiraga'|'wirama', 'severity': 0-100, 'message': str,
        'occurrences': [{'start_time', 'end_time', 'severity'}, ...]}, ...]
    """
    feedback = []
    classifier = MoveClassifier()
    move_mask = classifier.get_mask_for_move(selected_movement)
    ACTIVE_JOINTS = [j for j, w in enumerate(move_mask) if w > 0.05]

    joint_names = {
        8: 'left knee', 9: 'right knee',
        6: 'left hip', 7: 'right hip',
        2: 'left shoulder', 3: 'right shoulder',
        10: 'left ankle', 11: 'right ankle',
        12: 'spine',
    }

    # Wiraga: posture/form aggregated across entire video
    try:
        for j in ACTIVE_JOINTS:
            if j not in joint_names:
                continue
            s_mean = float(np.mean(student_data[:, j]))
            t_mean = float(np.mean(teacher_data[:, j]))
            diff = s_mean - t_mean
            if abs(diff) < 8:
                continue
            occurrences = _find_joint_occurrences(
                student_data, teacher_data, j, student_ts, teacher_ts
            )
            if not occurrences:
                continue
            severity = min(100, abs(diff) * 2)
            feedback.append({
                'type': 'wiraga',
                'severity': round(severity, 1),
                'joint_index': j,
                'message': _wiraga_message(joint_names[j], diff, selected_movement),
                'occurrences': occurrences,
            })
    except Exception as e:
        print(f'[feedback] wiraga pass error: {e}')

    # Wirama: grounded in music beats extracted from the student's own audio.
    # Only fires when beat detection produced enough events; suppressed otherwise
    # to avoid false positives from velocity-envelope comparison.
    _beats = student_beats if student_beats is not None else np.array([])
    if student_vel is not None and len(_beats) >= 4:
        try:
            miss_rate, occurrences = _find_missed_beats(
                student_vel, student_ts, _beats, move_mask
            )
            if miss_rate > 0.25 and occurrences:
                severity = min(100.0, miss_rate * 200)
                pct = int(miss_rate * 100)
                message = (
                    f'You are missing {pct}% of the beats. '
                    'Focus on landing each movement impulse exactly on the music.'
                    if miss_rate > 0.5 else
                    f'You are off-beat on about {pct}% of moments. '
                    'Try to sync your movement peaks with the music.'
                )
                feedback.append({
                    'type': 'wirama',
                    'severity': round(severity, 1),
                    'miss_rate': round(miss_rate, 3),
                    'message': message,
                    'occurrences': occurrences,
                })
        except Exception as e:
            print(f'[feedback] wirama pass error: {e}')
    elif len(_beats) < 4:
        print(f'[feedback] wirama skipped: only {len(_beats)} audio beats detected')

    # Power consistency: velocity variance across reps (no specific moments)
    if student_vel is not None:
        try:
            masked_vel = (student_vel * move_mask).sum(axis=1)
            if len(masked_vel) > 10:
                cv = float(np.std(masked_vel) / (np.mean(masked_vel) + 1e-6))
                if cv < 0.3:
                    feedback.append({
                        'type': 'wiraga',
                        'severity': 60,
                        'message': 'Your movements lack dynamic contrast. Add more power to your hits.',
                        'occurrences': [],
                    })
        except Exception as e:
            print(f'[feedback] power pass error: {e}')

    # Movement-specific wiraga messages from time-series analysis
    if wiraga_breakdown:
        comp = wiraga_breakdown.get('components', {})
        movement = wiraga_breakdown.get('movement_type', selected_movement)
        try:
            if movement == 'bouncing' and 'turning_points_count' in comp:
                tp  = comp['turning_points_count']
                suc = comp['successful_bends']
                if tp > 0 and suc / tp < 0.7:
                    feedback.append({
                        'type': 'wiraga',
                        'severity': round((1 - suc / tp) * 100, 1),
                        'message': (
                            'Your bounces are not deep enough. '
                            'Bend your knees to at least 90° at the bottom of each bounce.'
                        ),
                        'occurrences': [],
                    })

            elif movement == 'stepping' and 'strike_accuracy' in comp:
                sa  = comp['strike_accuracy']
                ws  = comp['weight_shift_score']
                if sa < 60:
                    feedback.append({
                        'type': 'wiraga',
                        'severity': round(100 - sa, 1),
                        'message': (
                            'Your step rhythm is unclear. '
                            'Plant your feet firmly on each beat.'
                        ),
                        'occurrences': [],
                    })
                if ws < 60:
                    feedback.append({
                        'type': 'wiraga',
                        'severity': round(100 - ws, 1),
                        'message': (
                            'Transfer your weight onto each stepping leg — '
                            'your hips should shift with your foot.'
                        ),
                        'occurrences': [],
                    })

            elif movement == 'sliding' and 'linearity' in comp:
                lin = comp['linearity']
                stb = comp['hip_stability']
                rch = comp['reach_ratio']
                if lin < 60:
                    feedback.append({
                        'type': 'wiraga',
                        'severity': round(100 - lin, 1),
                        'message': (
                            'Your slide is choppy. '
                            'Glide smoothly without lifting or jumping your feet.'
                        ),
                        'occurrences': [],
                    })
                if stb < 60:
                    feedback.append({
                        'type': 'wiraga',
                        'severity': round(100 - stb, 1),
                        'message': (
                            'Keep your upper body steady. '
                            'Your hips should not bounce up and down during the slide.'
                        ),
                        'occurrences': [],
                    })
                if rch < 60:
                    feedback.append({
                        'type': 'wiraga',
                        'severity': round(100 - rch, 1),
                        'message': 'Slide farther — at least shoulder-width apart.',
                        'occurrences': [],
                    })
        except Exception as e:
            print(f'[feedback] movement-specific pass error: {e}')

    feedback.sort(key=lambda x: x['severity'], reverse=True)
    return feedback




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
                        ['ffmpeg', '-y',
                         '-i', out_path,
                         '-ss', str(teacher_start), '-i', teacher_path,
                         '-map', '0:v', '-map', '1:a?',
                         '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26',
                         '-c:a', 'aac', '-b:a', '128k',
                         '-shortest',
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
            "[tv][sv]hstack=inputs=2[out];"
            f"[0:a]atrim=start={teacher_start},asetpts=PTS-STARTPTS[outa]"
        )
        cmd = (['ffmpeg', '-y',
                '-i', teacher_path,
                '-i', student_path]
               + dur_flag
               + ['-filter_complex', filter_g,
                  '-map', '[out]',
                  '-map', '[outa]',
                  '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26',
                  '-c:a', 'aac', '-b:a', '128k',
                  '-shortest',
                  '-movflags', '+faststart', out_path])
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0:
            return True
        # Retry without audio if teacher has no audio stream
        print("[compose] FFmpeg with audio failed, retrying without audio:",
              result.stderr.decode('utf-8', errors='replace')[-300:])
        filter_g_noaudio = (
            f"[0:v]trim=start={teacher_start},setpts=PTS-STARTPTS,fps=30,scale=-2:480,setsar=1,format=yuv420p[tv];"
            f"[1:v]trim=start={student_start},setpts=PTS-STARTPTS,fps=30,scale=-2:480,setsar=1,format=yuv420p[sv];"
            "[tv][sv]hstack=inputs=2[out]"
        )
        cmd2 = (['ffmpeg', '-y',
                 '-i', teacher_path,
                 '-i', student_path]
                + dur_flag
                + ['-filter_complex', filter_g_noaudio,
                   '-map', '[out]',
                   '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26',
                   '-movflags', '+faststart', '-an', out_path])
        result2 = subprocess.run(cmd2, capture_output=True, timeout=120)
        if result2.returncode != 0:
            print("[compose] FFmpeg fallback also failed:", result2.stderr.decode('utf-8', errors='replace')[-200:])
        return result2.returncode == 0
    except Exception as e:
        print(f"[compose] FFmpeg fallback error: {e}")
        return False


@app.route('/')
@login_required
def dashboard():
    return render_template('index.html', user=current_user)

@app.route('/test')
def test():
    """Test route"""
    return {'status': 'OK', 'message': 'Flask is running!'}


@app.route('/api/upload-student', methods=['POST'])
@login_required
def upload_student():
    """Handle student video upload and grading"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    selected_movement = request.form.get('movement_type', 'bouncing')
    selected_movement = selected_movement.strip().lower() if selected_movement else 'bouncing'

    VALID_MOVEMENTS = ('bouncing', 'stepping', 'sliding')
    if selected_movement not in VALID_MOVEMENTS:
        print(f'[warn] Invalid movement_type {selected_movement}, defaulting to bouncing')
        selected_movement = 'bouncing'

    # Validate reference exists
    if not has_reference(selected_movement):
        return jsonify({
            'error': f'Reference for "{selected_movement}" not available. '
                     'Please contact administrator to add reference video.'
        }), 400

    reference = get_reference(selected_movement)
    teacher_data      = reference['joint_angles'].copy()
    teacher_ts        = reference['timestamps'].copy()
    teacher_landmarks = reference.get('landmarks')          # None for old caches
    teacher_video_path_raw = os.path.join('assets', 'references',
                                          reference['video_filename'])

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

        # ── Idle-period trimming (Option B: auto, Option C: manual override) ─
        _s_vel_pre, _ = compute_joint_kinematics(np.array(student_data, dtype=float), student_ts)
        manual_start_raw = 0.0
        try:
            manual_start_raw = float(request.form.get('start_offset') or 0)
        except (ValueError, TypeError):
            pass

        if manual_start_raw > 0:
            # Option C: user marked start in raw video → subtract audio trim already applied
            extra_sec  = max(0.0, manual_start_raw - s_trim_sec)
            trim_source = 'manual'
        else:
            # Option B: auto-detect first sustained movement
            extra_sec  = trim_leading_idle(student_data, student_ts, _s_vel_pre) if _s_vel_pre is not None else 0.0
            trim_source = 'auto'

        if extra_sec > 0.05 and _s_vel_pre is not None:
            fps_s = len(student_ts) / float(student_ts[-1]) if student_ts[-1] > 0 else 30.0
            extra_frames = min(int(extra_sec * fps_s), len(student_data) - 3)
            if extra_frames > 0:
                student_data = student_data[extra_frames:]
                student_ts   = student_ts[extra_frames:]
                s_trim_sec  += extra_sec
                print(f'[idle-trim] source={trim_source}  +{extra_sec:.2f}s  total_s_trim={s_trim_sec:.2f}s')
        # ─────────────────────────────────────────────────────────────────────

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
        
        # Calculate scores using HipHopDanceScorer (Timing + Movement + Power)
        scorer = HipHopDanceScorer()
        scoring_results = scorer.evaluate(
            student_data, student_ts,
            teacher_data, teacher_ts,
            student_landmarks=landmarks_data,
            teacher_landmarks=teacher_landmarks,
            student_video_path=filepath,
            teacher_video_path=teacher_video_path_raw if os.path.exists(teacher_video_path_raw) else None,
            selected_movement=selected_movement,
        )

        overall_score = scoring_results['overall_score']
        arm_score   = scoring_results['regional']['arms']
        leg_score   = scoring_results['regional']['legs']
        torso_score = scoring_results['regional']['torso']

        # Reuse kinematics already computed by the scorer for feedback generation.
        _s_vel           = scoring_results.get('_s_vel')
        _t_vel           = scoring_results.get('_t_vel')
        _s_beats         = scoring_results.get('_s_beats', np.array([]))
        _wiraga_breakdown = scoring_results.get('wiraga_breakdown', {})

        # Generate aggregated feedback across all repetitions
        generalized = generate_generalized_feedback(
            student_data, student_ts, teacher_data, teacher_ts,
            _s_vel, _t_vel,
            selected_movement,
            student_beats=_s_beats,
            wiraga_breakdown=_wiraga_breakdown,
        )

        # Render the aligned side-by-side video on the server
        composed_name = f"composed_{student_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        composed_path = os.path.join(app.static_folder, 'uploads_videos', composed_name)

        if os.path.exists(teacher_video_path_raw):
            success = compose_side_by_side(
                teacher_video_path_raw, filepath, composed_path,
                teacher_start=t_trim_sec, student_start=s_trim_sec,
                fast_mode=True
            )
            composed_full_url = f"/static/uploads_videos/{composed_name}" if success else None
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
            # Three-pillar breakdown (new)
            'timing_score':   scoring_results.get('timing_score',   0),
            'movement_score': scoring_results.get('movement_score', 0),
            'power_score':    scoring_results.get('power_score',    0),
            'power_weight':   scoring_results.get('power_weight',   0),
            # Regional breakdown (from movement quality)
            'arm_score':   arm_score,
            'leg_score':   leg_score,
            'torso_score': torso_score,
            'feedback': feedback,
            'star_rating': star_rating,
            'timestamp': datetime.now().isoformat(),
            'composed_video': composed_full_url,
            'audio_offset_sec': round(audio_offset, 3),
            'teacher_trim_sec': round(t_trim_sec, 3),
            'student_trim_sec': round(s_trim_sec, 3),
            'scoring_detail': scoring_results.get('detail', {}),
            'move_timeline': scoring_results.get('move_timeline', []),
            'selected_movement': scoring_results.get('selected_movement', selected_movement),
        }
        
        results['generalized_feedback'] = generalized
        results['wiraga_breakdown']     = _wiraga_breakdown
        # Reference video served from /assets/references/
        results['teacher_video'] = f"/assets/references/{reference['video_filename']}"
        results['student_video'] = student_video_url if 'student_video_url' not in locals() else student_video_url
        results['teacher_landmarks_url'] = None
        results['student_landmarks_url'] = f"/static/uploads_videos/student_{student_name}_landmarks.json"

        # Produce prioritized semantic feedback for Arms, Legs, Torso
        try:
            min_len = min(len(student_data), len(teacher_data))
            diffs = np.abs(student_data[:min_len] - teacher_data[:min_len])
            mean_diffs = np.nanmean(diffs, axis=0)
            arm_idx = int(np.argmax(mean_diffs[0:6]))
            arm_label_map = ['L_Elbow', 'R_Elbow', 'L_Shoulder', 'R_Shoulder', 'L_Wrist', 'R_Wrist']
            arm_msg = get_semantic_feedback(arm_label_map[arm_idx],
                float(np.mean(student_data[:, arm_idx])), float(np.mean(teacher_data[:, arm_idx])))
            leg_offset = int(6 + np.argmax(mean_diffs[6:12]))
            leg_label_map = ['L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle']
            leg_msg = get_semantic_feedback(leg_label_map[leg_offset - 6],
                float(np.mean(student_data[:, leg_offset])), float(np.mean(teacher_data[:, leg_offset])))
            torso_msg = get_semantic_feedback('Spine',
                float(np.mean(student_data[:, 12])), float(np.mean(teacher_data[:, 12])))

            # Augment with timing/power notes from scorer
            timing_note = ''
            power_note  = ''
            t_detail = scoring_results.get('detail', {}).get('timing', {})
            p_detail = scoring_results.get('detail', {}).get('power', {})
            if t_detail.get('timing_score', 100) < 60:
                timing_note = 'Focus on hitting your moves exactly on the beat — your timing is drifting.'
            if p_detail.get('sharpness_score', 100) < 60:
                power_note = 'Your hits need more power — make them sharper and more explosive.'

            results['semantic_feedback'] = {
                'arm': arm_msg, 'leg': leg_msg, 'torso': torso_msg,
                'timing': timing_note, 'power': power_note,
            }
        except Exception:
            results['semantic_feedback'] = {'arm': '', 'leg': '', 'torso': '', 'timing': '', 'power': ''}

        # Save to database
        def _json_default(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            raise TypeError(f'Not serializable: {type(obj)}')

        try:
            _movement_score = float(results.get('movement_score', 0))
            _timing_score   = float(results.get('timing_score', 0))
            attempt = Attempt(
                user_id=current_user.id,
                movement_type=selected_movement,
                overall_score=overall_score,
                wiraga_score=round(_movement_score, 2),
                wirama_score=round(_timing_score, 2),
                video_path=results.get('student_video'),
                composed_path=composed_full_url,
                feedback_json=json.dumps({
                    'student_name':       student_name,
                    'frames':             len(student_data),
                    'feedback':           feedback,
                    'star_rating':        star_rating,
                    'generalized_feedback': generalized,
                    'semantic_feedback':  results.get('semantic_feedback', {}),
                    'teacher_video':      results.get('teacher_video'),
                    'audio_offset_sec':   round(audio_offset, 3),
                    'teacher_trim_sec':   round(t_trim_sec, 3),
                    'student_trim_sec':   round(s_trim_sec, 3),
                    'move_timeline':      scoring_results.get('move_timeline', []),
                    'selected_movement':  selected_movement,
                }, default=_json_default)
            )
            db.session.add(attempt)
            db.session.commit()
            results['attempt_id'] = attempt.id
        except Exception as db_err:
            print(f'[db] Failed to save Attempt: {db_err}')

        # Clean up
        os.remove(filepath)

        return jsonify(results)
    
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

@app.route('/api/history')
@login_required
def get_history():
    """Get grading history for current user from database."""
    try:
        attempts = (Attempt.query
                    .filter_by(user_id=current_user.id)
                    .order_by(Attempt.created_at.desc())
                    .all())
        history = []
        for a in attempts:
            extra = json.loads(a.feedback_json) if a.feedback_json else {}
            history.append({
                'id':               a.id,
                'student_name':     extra.get('student_name', ''),
                'movement_type':    a.movement_type,
                'selected_movement': a.movement_type,
                'overall_score':    a.overall_score,
                'timing_score':     a.wirama_score,
                'movement_score':   a.wiraga_score,
                'composed_video':   a.composed_path,
                'student_video':    a.video_path,
                'teacher_video':    extra.get('teacher_video'),
                'generalized_feedback': extra.get('generalized_feedback', extra.get('detailed_feedback', [])),
                'feedback':         extra.get('feedback', ''),
                'star_rating':      extra.get('star_rating', 0),
                'timestamp':        a.created_at.isoformat(),
                'frames':           extra.get('frames', 0),
                'semantic_feedback': extra.get('semantic_feedback', {}),
                'audio_offset_sec': extra.get('audio_offset_sec', 0),
                'teacher_trim_sec': extra.get('teacher_trim_sec', 0),
                'student_trim_sec': extra.get('student_trim_sec', 0),
                'move_timeline':    extra.get('move_timeline', []),
            })
        return jsonify(history)
    except Exception as e:
        print(f'[history] DB error: {e}')
        return jsonify([])

@app.route('/api/clear-history', methods=['POST'])
@login_required
def clear_history():
    """Delete all Attempt records for current user and their associated files."""
    try:
        attempts = Attempt.query.filter_by(user_id=current_user.id).all()
        deleted_files = 0

        for a in attempts:
            # Delete composed video and student video files if they exist on disk
            for url in [a.composed_path, a.video_path]:
                if not url:
                    continue
                # Convert URL path like /static/uploads_videos/foo.mp4 → absolute path
                rel = url.lstrip('/')
                abs_path = os.path.join(os.path.dirname(__file__), rel)
                if os.path.isfile(abs_path):
                    try:
                        os.remove(abs_path)
                        deleted_files += 1
                    except Exception:
                        pass
            db.session.delete(a)

        db.session.commit()

        # Also clean up temp CSV/landmarks/thumbs left in uploads/
        for pattern in ['student_*.csv', 'student_*_landmarks.json']:
            for p in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], pattern)):
                try: os.remove(p)
                except Exception: pass

        return jsonify({'success': True, 'deleted_files': deleted_files})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
@login_required
def get_status():
    """Get system status."""
    return jsonify({
        'references_available': list_available_references(),
        'student_attempts': current_user.get_attempt_count(),
        'student_attempts_by_movement': {
            m: current_user.get_attempt_count(m)
            for m in ['bouncing', 'stepping', 'sliding']
        },
    })


@app.route('/api/references')
@login_required
def get_references():
    """Return metadata for all available reference videos."""
    refs = get_all_references_meta()
    result = {}
    for movement, meta in refs.items():
        result[movement] = {
            **meta,
            'video_url': f'/assets/references/{meta["video_filename"]}',
            'thumb_url': f'/assets/references/{movement}.thumb.jpg',
        }
    return jsonify(result)


@app.route('/assets/references/<path:filename>')
@login_required
def serve_reference(filename):
    """Serve reference videos and thumbnails."""
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), 'assets', 'references'),
        filename
    )


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)