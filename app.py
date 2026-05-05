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

    Move archetypes and the joints that MATTER for them:
      bounce      — rhythmic up/down: hips, knees, spine only
      arm_wave    — fluid arm propagation: elbows, shoulders, wrists
      footwork    — foot/leg patterns: knees, ankles, hips
      chest_pop   — chest isolation: spine, shoulders only
      full_body   — everything counts (default / catch-all)
      freeze      — static hold: all joints, extra weight on spine
      groove      — subtle rock: hips, knees, spine (less strict than bounce)
    """

    # mask values: 1.0 = scored normally, 0.05 = nearly ignored, 0.0 = skip entirely
    MOVE_MASKS: dict = {
        # idx:        0     1     2     3     4     5     6     7     8     9    10    11    12
        #          L_Elb R_Elb L_Sho R_Sho L_Wri R_Wri L_Hip R_Hip L_Kne R_Kne L_Ank R_Ank Spin
        'bounce':     [0.05, 0.05, 0.05, 0.05, 0.00, 0.00, 1.20, 1.20, 1.20, 1.20, 0.05, 0.05, 1.50],
        'groove':     [0.10, 0.10, 0.20, 0.20, 0.00, 0.00, 1.20, 1.20, 1.00, 1.00, 0.10, 0.10, 1.20],
        'arm_wave':   [1.20, 1.20, 1.20, 1.20, 1.00, 1.00, 0.05, 0.05, 0.05, 0.05, 0.00, 0.00, 0.30],
        'chest_pop':  [0.20, 0.20, 1.20, 1.20, 0.10, 0.10, 0.30, 0.30, 0.10, 0.10, 0.00, 0.00, 1.50],
        'footwork':   [0.05, 0.05, 0.05, 0.05, 0.00, 0.00, 1.00, 1.00, 1.30, 1.30, 1.20, 1.20, 0.30],
        'freeze':     [0.80, 0.80, 1.00, 1.00, 0.40, 0.40, 1.00, 1.00, 1.00, 1.00, 0.60, 0.60, 1.50],
        'top_rock':   [0.80, 0.80, 1.00, 1.00, 0.50, 0.50, 1.20, 1.20, 0.60, 0.60, 0.10, 0.10, 1.20],
        'body_roll':  [0.30, 0.30, 0.80, 0.80, 0.20, 0.20, 1.00, 1.00, 0.50, 0.50, 0.10, 0.10, 1.50],
        'popping':    [1.00, 1.00, 1.20, 1.20, 0.80, 0.80, 0.80, 0.80, 0.60, 0.60, 0.30, 0.30, 1.20],
        'locking':    [1.20, 1.20, 1.00, 1.00, 1.00, 1.00, 0.80, 0.80, 0.80, 0.80, 0.50, 0.50, 0.80],
        'full_body':  [0.70, 0.70, 1.00, 1.00, 0.30, 0.30, 1.20, 1.20, 1.10, 1.10, 0.30, 0.30, 1.50],
    }

    # ── Per-move scoring profiles ─────────────────────────────────────────────
    # Each profile defines:
    #   pillar_weights  — how much Timing / Movement / Power contribute to overall score
    #   focus_joints    — human-readable list of what's being judged (for UI)
    #   display_name    — friendly label shown to the user
    #   description     — one-line explanation of what the move emphasises
    #   feedback_cues   — what good execution looks and feels like
    MOVE_PROFILES: dict = {
        'bounce': {
            'display_name': 'Bounce',
            'description':  'Rhythmic up-down motion driven by hips and knees',
            'pillar_weights': {'timing': 0.45, 'movement': 0.30, 'power': 0.25},
            'focus_joints':  ['Hips', 'Knees', 'Spine'],
            'ignored_joints': ['Wrists', 'Ankles'],
            'feedback_cues': 'Knees should absorb and rebound on every beat. '
                             'Hips drive the bounce — spine follows naturally.',
        },
        'groove': {
            'display_name': 'Groove',
            'description':  'Subtle rhythmic sway — the foundation of all hip hop',
            'pillar_weights': {'timing': 0.50, 'movement': 0.35, 'power': 0.15},
            'focus_joints':  ['Hips', 'Knees', 'Spine'],
            'ignored_joints': ['Wrists', 'Ankles'],
            'feedback_cues': 'Movement should feel effortless and musical. '
                             'Less is more — every micro-shift should land on the beat.',
        },
        'arm_wave': {
            'display_name': 'Arm Wave',
            'description':  'Fluid sequential wave propagating through the arm',
            'pillar_weights': {'timing': 0.30, 'movement': 0.50, 'power': 0.20},
            'focus_joints':  ['Shoulders', 'Elbows', 'Wrists'],
            'ignored_joints': ['Knees', 'Ankles'],
            'feedback_cues': 'Wave must pass through shoulder → elbow → wrist in sequence. '
                             'Each joint isolates cleanly before passing to the next.',
        },
        'chest_pop': {
            'display_name': 'Chest Pop',
            'description':  'Sharp chest isolation — a core popping/locking technique',
            'pillar_weights': {'timing': 0.35, 'movement': 0.30, 'power': 0.35},
            'focus_joints':  ['Shoulders', 'Spine'],
            'ignored_joints': ['Knees', 'Ankles', 'Wrists'],
            'feedback_cues': 'Chest leads the hit — shoulders follow. '
                             'Lower body stays still. Pop must be sharp, not gradual.',
        },
        'footwork': {
            'display_name': 'Footwork',
            'description':  'Fast foot and leg patterns — b-boy / house footwork',
            'pillar_weights': {'timing': 0.40, 'movement': 0.35, 'power': 0.25},
            'focus_joints':  ['Ankles', 'Knees', 'Hips'],
            'ignored_joints': ['Wrists', 'Elbows'],
            'feedback_cues': 'Foot placement must be precise and rhythmic. '
                             'Weight transfers should be clean — no dragging.',
        },
        'freeze': {
            'display_name': 'Freeze',
            'description':  'Static hold — body locked in a pose with zero drift',
            'pillar_weights': {'timing': 0.20, 'movement': 0.30, 'power': 0.50},
            'focus_joints':  ['All joints'],
            'ignored_joints': [],
            'feedback_cues': 'Every joint must hold position. '
                             'Any wobble or drift is penalised. Core and spine are critical.',
        },
        'top_rock': {
            'display_name': 'Top Rock',
            'description':  'Standing b-boy/b-girl entry steps — upper body + hip driven',
            'pillar_weights': {'timing': 0.40, 'movement': 0.35, 'power': 0.25},
            'focus_joints':  ['Hips', 'Shoulders', 'Elbows', 'Spine'],
            'ignored_joints': ['Ankles', 'Wrists'],
            'feedback_cues': 'Arms and hips must coordinate. '
                             'Step rhythm drives the whole movement.',
        },
        'body_roll': {
            'display_name': 'Body Roll',
            'description':  'Sequential wave from chest through hips — fluid and continuous',
            'pillar_weights': {'timing': 0.30, 'movement': 0.55, 'power': 0.15},
            'focus_joints':  ['Spine', 'Hips', 'Shoulders'],
            'ignored_joints': ['Ankles', 'Wrists'],
            'feedback_cues': 'Motion must be sequential and smooth — no jerky segments. '
                             'Spine is the conductor of the whole wave.',
        },
        'popping': {
            'display_name': 'Popping',
            'description':  'Muscle contractions creating sharp hits across the body',
            'pillar_weights': {'timing': 0.35, 'movement': 0.25, 'power': 0.40},
            'focus_joints':  ['Shoulders', 'Elbows', 'Hips', 'Spine'],
            'ignored_joints': ['Ankles'],
            'feedback_cues': 'Each pop is a sharp contraction then full release. '
                             'No telegraphing — hit must be instant.',
        },
        'locking': {
            'display_name': 'Locking',
            'description':  'Exaggerated arm locks with funky rhythm and character',
            'pillar_weights': {'timing': 0.40, 'movement': 0.30, 'power': 0.30},
            'focus_joints':  ['Elbows', 'Wrists', 'Shoulders'],
            'ignored_joints': ['Ankles'],
            'feedback_cues': 'Lock must be crisp and held for exactly the right duration. '
                             'Character and attitude are part of the execution.',
        },
        'full_body': {
            'display_name': 'Full Routine',
            'description':  'Complete routine — all joints scored by move context',
            'pillar_weights': {'timing': 0.35, 'movement': 0.35, 'power': 0.30},
            'focus_joints':  ['All joints'],
            'ignored_joints': [],
            'feedback_cues': 'Every part of the body is evaluated. '
                             'Move classifier adapts weights automatically per section.',
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

        Heuristic rules derived from Hip Hop biomechanics:
          - bounce:    high knee/hip velocity, low arm velocity, oscillatory pattern
          - arm_wave:  high elbow/shoulder/wrist velocity, low leg velocity
          - chest_pop: high shoulder+spine velocity, moderate elbow, low legs
          - footwork:  high ankle/knee velocity, low arm velocity
          - freeze:    near-zero velocity everywhere
          - groove:    moderate hip/knee, low arms — less explosive than bounce
          - full_body: everything above average
        """
        if vel_segment is None or len(vel_segment) < 2:
            return 'full_body'

        mean_vel = vel_segment.mean(axis=0)   # shape (13,)

        arm_vel   = mean_vel[[0,1,2,3,4,5]].mean()    # elbows, shoulders, wrists
        leg_vel   = mean_vel[[6,7,8,9,10,11]].mean()  # hips, knees, ankles
        hip_vel   = mean_vel[[6,7]].mean()
        knee_vel  = mean_vel[[8,9]].mean()
        ankle_vel = mean_vel[[10,11]].mean()
        sho_vel   = mean_vel[[2,3]].mean()
        spine_vel = mean_vel[12]
        total_vel = mean_vel.mean()

        # Freeze: almost nothing moving
        if total_vel < 5.0:
            return 'freeze'

        # Arm wave: arms dominate, legs quiet
        if arm_vel > leg_vel * 2.0 and arm_vel > 20:
            return 'arm_wave'

        # Footwork: ankles/knees high, arms low
        if ankle_vel > arm_vel * 1.5 and ankle_vel > 15:
            return 'footwork'

        # Chest pop: shoulders + spine spike, legs quiet
        if sho_vel > arm_vel * 0.8 and spine_vel > leg_vel * 1.2 and leg_vel < 20:
            return 'chest_pop'

        # Bounce vs groove — both leg-dominant, differentiate by intensity
        if hip_vel > arm_vel * 1.2 or knee_vel > arm_vel * 1.2:
            if total_vel > 25:
                return 'bounce'    # explosive, high velocity
            else:
                return 'groove'    # subtle rhythmic sway

        # Default
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
                     student_beats: np.ndarray, teacher_beats: np.ndarray) -> dict:
        """
        Measure how well movement peaks align with musical beats.

        For each beat in the teacher recording we find the nearest teacher
        velocity spike; we then check whether the student has a spike within
        ±WINDOW_SEC of the corresponding student beat.  Score = hit-rate × 100.

        Falls back to cross-correlation of velocity envelopes when beat
        detection yields too few events (e.g. no audio).
        """
        WINDOW_SEC = 0.18   # ±180 ms tolerance — tighter than DTW stretch

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
                                student_vel: np.ndarray, teacher_vel: np.ndarray) -> dict:
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
        """
        student_data = np.array(student_data, dtype=float)
        teacher_data = np.array(teacher_data, dtype=float)

        if len(student_data) < 2 or len(teacher_data) < 2:
            return {'movement_score': 0, 'regional': {r: 0 for r in self.REGIONS},
                    'velocity_score': 0, 'pose_shape_score': 0}

        # ── Build per-frame move masks from TEACHER velocity (ground truth move) ──
        # We use the teacher's movement to define what type of move is happening,
        # then apply that mask to both teacher and student so irrelevant joints
        # (e.g. wrists during a bounce) don't affect the score.
        t_masks = self.classifier.get_frame_masks(teacher_vel)  # shape (T, 13)
        # Pad or trim to match teacher_data length
        n_t = len(teacher_data)
        if len(t_masks) < n_t:
            t_masks = np.vstack([t_masks, np.tile(t_masks[-1], (n_t - len(t_masks), 1))])
        else:
            t_masks = t_masks[:n_t]

        # ── A. Velocity pattern DTW (move-context-aware) ──
        # Frame index tracker so we can look up the teacher mask at DTW warp position j
        def _vel_dist(a, b):
            # a = student frame velocity (13,), b = teacher frame velocity (13,)
            # Use fallback JOINT_WEIGHTS here — mask is applied in post-scoring
            diff = (a - b) * self.JOINT_WEIGHTS
            return float(np.sqrt((diff ** 2).sum()))

        try:
            v_dist, v_path = fastdtw(student_vel, teacher_vel, dist=_vel_dist)
            # Re-weight path errors by teacher move mask
            masked_v_errors = []
            for i, j in v_path:
                si = min(i, len(student_vel) - 1)
                tj = min(j, len(teacher_vel) - 1)
                diff = np.abs(student_vel[si] - teacher_vel[tj])
                mask = self.classifier.get_mask_for_move(
                    self.classifier.classify_segment(teacher_vel[max(0,tj-5):tj+5])
                )
                masked_v_errors.append(float((diff * mask).sum()))
            v_avg = float(np.mean(masked_v_errors)) if masked_v_errors else 1.0
            velocity_score = max(0.0, min(100.0, (1 - v_avg / 80.0) * 100))
        except Exception:
            velocity_score = 50.0
            v_path = []

        # ── B. Normalised angle DTW (move-context-aware) ──
        def _angle_dist(a, b):
            norm_a = a / self.MAX_ROM
            norm_b = b / self.MAX_ROM
            diff = (norm_a - norm_b) * self.JOINT_WEIGHTS
            return float(np.sqrt((diff ** 2).sum()))

        try:
            a_dist, a_path = fastdtw(student_data, teacher_data, dist=_angle_dist)
            a_avg = a_dist / max(len(a_path), 1)
            pose_shape_score = max(0.0, min(100.0, (1 - a_avg) * 100))
        except Exception:
            pose_shape_score = 50.0
            a_path = []

        movement_score = 0.55 * velocity_score + 0.45 * pose_shape_score

        # ── Regional breakdown using angle path + move masks ──
        # Each frame's error is weighted by the teacher's move mask so regions
        # that are irrelevant for that move type contribute less to the score.
        regional = {}
        if a_path:
            s_al  = np.array([student_data[i]  for i, _ in a_path], dtype=float)
            t_al  = np.array([teacher_data[j]  for _, j in a_path], dtype=float)
            # Per-path-step move mask from teacher velocity
            path_masks = np.array([
                self.classifier.get_mask_for_move(
                    self.classifier.classify_segment(
                        teacher_vel[max(0, j-5): j+5] if j < len(teacher_vel) else teacher_vel[-5:]
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
                 student_video_path: str = None,
                 teacher_video_path: str = None) -> dict:
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
        t_res = self.score_timing(
            s_vel, student_ts[:len(s_vel)],
            t_vel, teacher_ts[:len(t_vel)],
            s_beats, t_beats,
        )
        m_res = self.score_movement_quality(
            student_data, student_ts,
            teacher_data, teacher_ts,
            s_vel, t_vel,
        )
        p_res = self.score_power(s_acc, t_acc, s_vel, t_vel)

        timing_score   = t_res['timing_score']
        movement_score = m_res['movement_score']
        power_score    = p_res['power_score']

        # Weighted composite — mirrors rubrik expert:
        #   Musicality 35% | Movement 35% | Power 30%
        overall_score = (
            0.35 * timing_score +
            0.35 * movement_score +
            0.30 * power_score
        )
        overall_score = round(max(0.0, min(100.0, overall_score)), 2)

        return {
            'overall_score': overall_score,
            'timing_score':  round(timing_score,   2),
            'movement_score': round(movement_score, 2),
            'power_score':   round(power_score,     2),
            # regional breakdown comes from movement quality sub-scores
            'regional': m_res['regional'],
            'detail': {
                'timing':   t_res,
                'movement': m_res,
                'power':    p_res,
            },
            'move_timeline': move_timeline,
        }


def generate_detailed_feedback(student_data, student_ts, teacher_data, teacher_ts=None,
                               threshold_deg=25, top_n=None,
                               student_vel=None, teacher_vel=None):
    """Generate time-stamped, per-joint feedback using velocity-aware DTW alignment.

    Primary signal: velocity difference (how the movement flows), not raw angle.
    Secondary signal: sustained angle divergence (consistently wrong posture).

    Returns a list of feedback entries ordered by severity.
    Each entry: {joint, start_time, end_time, avg_diff, student_angle,
                 teacher_angle, velocity_issue, message, teacher_time}
    """
    joint_names = ['L_Elbow', 'R_Elbow', 'L_Shoulder', 'R_Shoulder',
                   'L_Wrist', 'R_Wrist',
                   'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee',
                   'L_Ankle', 'R_Ankle', 'Spine']

    # Per-joint angle-diff threshold multipliers (unchanged — still useful for posture check)
    joint_threshold_mult = {
        4: 1.3, 5: 1.3,
        6: 1.5, 7: 1.5,
        8: 1.4, 9: 1.4,
        10: 1.5, 11: 1.5,
    }

    feedback_list = []

    # ── A. Velocity-based feedback (primary) ─────────────────────────────────
    # Identify joints where student velocity pattern differs significantly.
    # Joints that are irrelevant for the current move type are SKIPPED entirely
    # (e.g. wrists during a bounce, ankles during a chest pop).
    _classifier = MoveClassifier()

    if student_vel is not None and teacher_vel is not None:
        try:
            min_len = min(len(student_vel), len(teacher_vel))
            vel_diff = student_vel[:min_len] - teacher_vel[:min_len]  # (N, 13)
            # Per-frame teacher move masks (use teacher as ground-truth move ref)
            t_frame_masks = _classifier.get_frame_masks(teacher_vel[:min_len])  # (N, 13)
            # Use student timestamps for timing
            s_ts_vel = student_ts[:min_len] if student_ts is not None else np.arange(min_len) / 30.0

            VEL_THRESHOLD = 30.0  # deg/s — significant velocity mismatch
            # A joint is considered ACTIVE for a frame if its mask weight > 0.08
            # (anything below that means the move type doesn't care about it)
            MASK_ACTIVE_THRESHOLD = 0.08
            MIN_DUR = 0.4

            for j in range(vel_diff.shape[1]):
                col = vel_diff[:, j]
                runs = []
                run = []
                for i, v in enumerate(col):
                    # Skip this joint at this frame if the move mask says it's irrelevant
                    if t_frame_masks[i, j] < MASK_ACTIVE_THRESHOLD:
                        if run:
                            runs.append(run)
                            run = []
                        continue
                    if abs(v) >= VEL_THRESHOLD:
                        run.append((float(s_ts_vel[i]), float(v),
                                    float(student_vel[i, j]), float(teacher_vel[i, j])))
                    else:
                        if run:
                            runs.append(run)
                            run = []
                if run:
                    runs.append(run)

                for run in runs:
                    times = [r[0] for r in run]
                    start_t, end_t = min(times), max(times)
                    if (end_t - start_t) < MIN_DUR:
                        continue
                    avg_vdiff = float(np.mean([abs(r[1]) for r in run]))
                    avg_s_vel = float(np.mean([r[2] for r in run]))
                    avg_t_vel = float(np.mean([r[3] for r in run]))

                    # Determine issue type
                    if avg_s_vel < avg_t_vel * 0.6:
                        vel_issue = 'too_slow'  # movement too soft / hesitant
                    elif avg_s_vel > avg_t_vel * 1.5:
                        vel_issue = 'too_fast'  # movement rushed / uncontrolled
                    else:
                        vel_issue = 'mistimed'  # present but wrong moment

                    feedback_list.append({
                        'joint': joint_names[j],
                        'start_time': round(start_t, 2),
                        'end_time': round(end_t, 2),
                        'avg_diff': round(avg_vdiff, 1),
                        'student_angle': None,
                        'teacher_angle': None,
                        'velocity_issue': vel_issue,
                        'source': 'velocity',
                        'teacher_time': None,
                    })
        except Exception as e:
            print(f'[feedback] velocity pass error: {e}')

    # ── B. Angle-based feedback (posture / sustained divergence) ─────────────
    try:
        distance, path = fastdtw(student_data, teacher_data, dist=euclidean)
        path_sorted = sorted(path, key=lambda p: p[0])

        per_joint_events = {j: [] for j in range(student_data.shape[1])}
        for s_idx, t_idx in path_sorted:
            if s_idx >= len(student_ts) or t_idx >= len(teacher_data):
                continue
            s_time = float(student_ts[s_idx])
            t_time = float(teacher_ts[t_idx]) if (teacher_ts is not None and t_idx < len(teacher_ts)) else None
            s_row = student_data[s_idx]
            t_row = teacher_data[t_idx]
            diffs = t_row - s_row
            for j in range(len(diffs)):
                per_joint_events[j].append((s_time, float(diffs[j]),
                                             float(s_row[j]), float(t_row[j]), t_time))

        MIN_DURATION_SEC = 0.6

        for j, events in per_joint_events.items():
            if not events:
                continue
            eff_threshold = threshold_deg * joint_threshold_mult.get(j, 1.0)
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

            for run in runs:
                times = [r[0] for r in run]
                start_t, end_t = min(times), max(times)
                if (end_t - start_t) < MIN_DURATION_SEC:
                    continue
                avg_diff = float(np.mean([r[1] for r in run]))
                avg_s = float(np.mean([r[2] for r in run]))
                avg_t = float(np.mean([r[3] for r in run]))
                t_times = [r[4] for r in run if r[4] is not None]
                teacher_time = float(np.mean(t_times)) if t_times else None

                feedback_list.append({
                    'joint': joint_names[j],
                    'start_time': round(start_t, 2),
                    'end_time': round(end_t, 2),
                    'avg_diff': round(abs(avg_diff), 1),
                    'student_angle': round(avg_s, 1),
                    'teacher_angle': round(avg_t, 1),
                    'velocity_issue': None,
                    'source': 'angle',
                    'teacher_time': teacher_time,
                })
    except Exception as e:
        print(f'[feedback] angle pass error: {e}')

    # Sort by severity (avg_diff) descending
    feedback_list.sort(key=lambda x: x['avg_diff'], reverse=True)
    return feedback_list if top_n is None else feedback_list[:top_n]


def _friendly_message(entry):
    joint = entry.get('joint', '')
    start = entry.get('start_time', 0.0)
    avg_diff = entry.get('avg_diff', 0.0)
    s_ang = entry.get('student_angle')
    t_ang = entry.get('teacher_angle')
    vel_issue = entry.get('velocity_issue')
    source = entry.get('source', 'angle')

    joint_name = joint.lower().replace('_', ' ')

    # ── Velocity-based message (movement feel / timing) ──
    if source == 'velocity' and vel_issue:
        if vel_issue == 'too_slow':
            return (f"At {start:.1f}s — Hit your {joint_name} harder and sharper. "
                    f"Your movement is too soft here — teacher's motion is {avg_diff:.0f} deg/s faster.")
        elif vel_issue == 'too_fast':
            return (f"At {start:.1f}s — Slow down and control your {joint_name}. "
                    f"You're moving {avg_diff:.0f} deg/s too fast — the movement feels rushed.")
        else:  # mistimed
            return (f"At {start:.1f}s — Your {joint_name} move is off-beat. "
                    f"Sync this movement tighter to the music.")

    # ── Angle-based message (posture / position) ──
    if s_ang is None or t_ang is None:
        return f"At {start:.1f}s — Adjust your {joint_name} position."

    increase = t_ang > s_ang
    if 'knee' in joint_name or 'elbow' in joint_name:
        verb = 'straighten' if increase else 'bend'
    elif 'shoulder' in joint_name:
        verb = 'raise' if increase else 'lower'
    elif 'hip' in joint_name:
        verb = 'stand straighter at' if increase else 'bend deeper at'
    elif 'wrist' in joint_name:
        verb = 'extend' if increase else 'flex'
    elif 'ankle' in joint_name:
        verb = 'point your toes more at' if increase else 'flex your foot at'
    elif 'spine' in joint_name:
        verb = 'stand taller — straighten your back at' if increase else 'lean your torso forward at'
    else:
        verb = 'open up' if increase else 'tighten'

    return (f"At {start:.1f}s — {verb.capitalize()} your {joint_name} "
            f"(you: {s_ang:.0f}°, target: {t_ang:.0f}°, diff: {abs(avg_diff):.0f}°).")


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
        
        # Calculate scores using HipHopDanceScorer (Timing + Movement + Power)
        scorer = HipHopDanceScorer()
        scoring_results = scorer.evaluate(
            student_data, student_ts,
            teacher_data, teacher_ts,
            student_video_path=filepath,
            teacher_video_path=teacher_video_path_raw if os.path.exists(teacher_video_path_raw) else None,
        )

        overall_score = scoring_results['overall_score']
        arm_score   = scoring_results['regional']['arms']
        leg_score   = scoring_results['regional']['legs']
        torso_score = scoring_results['regional']['torso']

        # Derive kinematics for feedback generation
        _s_vel, _s_acc = compute_joint_kinematics(student_data, student_ts)
        _t_vel, _t_acc = compute_joint_kinematics(teacher_data, teacher_ts)

        # Generate detailed feedback (velocity-aware + angle-based)
        detailed = generate_detailed_feedback(
            student_data, student_ts, teacher_data, teacher_ts=teacher_ts,
            threshold_deg=20,
            student_vel=_s_vel, teacher_vel=_t_vel,
        )

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
            # Three-pillar breakdown (new)
            'timing_score':   scoring_results.get('timing_score',   0),
            'movement_score': scoring_results.get('movement_score', 0),
            'power_score':    scoring_results.get('power_score',    0),
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