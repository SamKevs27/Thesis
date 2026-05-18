"""
Generate placeholder reference videos for the 3 basic movements.

Produces animated stick-figure MP4s suitable for end-to-end pipeline
testing.  MediaPipe detection will be unreliable on these, but the files
themselves are valid and the precompute script will still produce cached
JSON (with extraction_status "degraded" if few landmarks are found).

Usage:
    python scripts/generate_placeholders.py
"""

import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR      = os.path.join(PROJECT_ROOT, 'assets', 'references')

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Video parameters
# ---------------------------------------------------------------------------
FPS    = 30
DURATION_SEC = 5.0
TOTAL_FRAMES = int(FPS * DURATION_SEC)
W, H   = 720, 1280          # portrait

# Colours
BG_COLOR    = (18, 18, 28)   # dark navy
BODY_COLOR  = (124, 106, 247)  # accent purple
HEAD_COLOR  = (165, 148, 249)
SHADOW_COLOR = (40, 40, 60)

# ---------------------------------------------------------------------------
# Stick-figure joint positions (normalised 0-1 in frame coordinates)
# ---------------------------------------------------------------------------
# Standing pose: head, neck, L/R shoulder, L/R elbow, L/R wrist,
#                hip_c, L/R hip, L/R knee, L/R ankle

def joints_standing(cx: float, cy_offset: float = 0.0) -> dict:
    """Return joint pixel coords for a neutral standing figure."""
    # cx in 0..W, cy_offset shifts the whole figure up (-) or down (+)
    base_y = H * 0.45 + cy_offset
    return {
        'head':       (cx,       base_y - 0.18 * H),
        'neck':       (cx,       base_y - 0.10 * H),
        'l_shoulder': (cx - 80,  base_y - 0.09 * H),
        'r_shoulder': (cx + 80,  base_y - 0.09 * H),
        'l_elbow':    (cx - 95,  base_y + 0.01 * H),
        'r_elbow':    (cx + 95,  base_y + 0.01 * H),
        'l_wrist':    (cx - 105, base_y + 0.10 * H),
        'r_wrist':    (cx + 105, base_y + 0.10 * H),
        'hip_c':      (cx,       base_y + 0.05 * H),
        'l_hip':      (cx - 50,  base_y + 0.06 * H),
        'r_hip':      (cx + 50,  base_y + 0.06 * H),
        'l_knee':     (cx - 55,  base_y + 0.18 * H),
        'r_knee':     (cx + 55,  base_y + 0.18 * H),
        'l_ankle':    (cx - 58,  base_y + 0.30 * H),
        'r_ankle':    (cx + 58,  base_y + 0.30 * H),
    }


def draw_figure(img: np.ndarray, j: dict, alpha: float = 1.0) -> None:
    """Draw stick figure onto img given joint dict."""
    def pt(name):
        x, y = j[name]
        return (int(round(x)), int(round(y)))

    def bc(color, a=alpha):
        return tuple(int(c * a + BG_COLOR[i] * (1 - a)) for i, c in enumerate(color))

    lw = 5
    connections = [
        ('head', 'neck'), ('neck', 'l_shoulder'), ('neck', 'r_shoulder'),
        ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'),
        ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'),
        ('neck', 'hip_c'),
        ('hip_c', 'l_hip'), ('hip_c', 'r_hip'),
        ('l_hip', 'l_knee'), ('l_knee', 'l_ankle'),
        ('r_hip', 'r_knee'), ('r_knee', 'r_ankle'),
    ]
    for a_name, b_name in connections:
        cv2.line(img, pt(a_name), pt(b_name), bc(BODY_COLOR), lw, cv2.LINE_AA)

    # Head circle
    hx, hy = pt('head')
    cv2.circle(img, (hx, hy), 28, bc(HEAD_COLOR), -1, cv2.LINE_AA)
    cv2.circle(img, (hx, hy), 28, bc(BODY_COLOR), 2,  cv2.LINE_AA)

    # Joints
    for name, pos in j.items():
        if name == 'head':
            continue
        cv2.circle(img, pt(name), 6, bc(HEAD_COLOR), -1, cv2.LINE_AA)


def label_frame(img: np.ndarray, text: str, frame_idx: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text.upper(), (30, 60), font, 1.4,
                (200, 200, 220), 2, cv2.LINE_AA)
    cv2.putText(img, f'frame {frame_idx:03d}', (30, 100), font, 0.6,
                (80, 80, 100), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Per-movement frame generators
# ---------------------------------------------------------------------------

def frame_bouncing(i: int) -> np.ndarray:
    """Knee-bend bounce at ~2 Hz."""
    img = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
    cx  = W / 2

    # Vertical position of whole figure oscillates
    bounce_phase = 2 * np.pi * 2.0 * i / FPS   # 2 bounces/sec
    cy_offset    = 60 * np.sin(bounce_phase)     # ±60 px

    j = joints_standing(cx, cy_offset)

    # Bend knees more when figure is at the bottom
    bend = max(0.0, -np.sin(bounce_phase)) * 0.12 * H
    j['l_knee'] = (j['l_knee'][0], j['l_knee'][1] + bend * 0.4)
    j['r_knee'] = (j['r_knee'][0], j['r_knee'][1] + bend * 0.4)
    j['l_ankle'] = (j['l_ankle'][0], j['l_ankle'][1])
    j['r_ankle'] = (j['r_ankle'][0], j['r_ankle'][1])

    draw_figure(img, j)
    label_frame(img, 'bouncing', i)
    return img


def frame_stepping(i: int) -> np.ndarray:
    """Alternating foot lift at ~1.5 Hz."""
    img = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
    cx  = W / 2

    phase = 2 * np.pi * 1.5 * i / FPS
    j = joints_standing(cx)

    # Left foot lifts when sin > 0, right when sin < 0
    if np.sin(phase) > 0:
        lift = np.sin(phase) * 0.08 * H
        j['l_knee']  = (j['l_knee'][0],  j['l_knee'][1]  - lift * 0.6)
        j['l_ankle'] = (j['l_ankle'][0], j['l_ankle'][1] - lift)
    else:
        lift = -np.sin(phase) * 0.08 * H
        j['r_knee']  = (j['r_knee'][0],  j['r_knee'][1]  - lift * 0.6)
        j['r_ankle'] = (j['r_ankle'][0], j['r_ankle'][1] - lift)

    draw_figure(img, j)
    label_frame(img, 'stepping', i)
    return img


def frame_sliding(i: int) -> np.ndarray:
    """Horizontal slide ±200 px at ~0.8 Hz."""
    img = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)

    phase = 2 * np.pi * 0.8 * i / FPS
    cx = W / 2 + 200 * np.sin(phase)

    j = joints_standing(cx)

    # Lean torso slightly in direction of travel
    lean = 25 * np.cos(phase)   # cos gives velocity direction
    j['neck']       = (j['neck'][0]       + lean * 0.3, j['neck'][1])
    j['l_shoulder'] = (j['l_shoulder'][0] + lean * 0.3, j['l_shoulder'][1])
    j['r_shoulder'] = (j['r_shoulder'][0] + lean * 0.3, j['r_shoulder'][1])

    draw_figure(img, j)
    label_frame(img, 'sliding', i)
    return img


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

GENERATORS = {
    'bouncing': frame_bouncing,
    'stepping': frame_stepping,
    'sliding':  frame_sliding,
}


def write_video(movement: str) -> None:
    out_path   = os.path.join(OUT_DIR, f'{movement}.mp4')
    thumb_path = os.path.join(OUT_DIR, f'{movement}.thumb.jpg')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))
    if not writer.isOpened():
        print(f'[ERROR] Cannot open VideoWriter for {out_path}', file=sys.stderr)
        return

    gen = GENERATORS[movement]
    first_frame = None
    for i in range(TOTAL_FRAMES):
        frame = gen(i)
        writer.write(frame)
        if i == 0:
            first_frame = frame.copy()

    writer.release()

    if first_frame is not None:
        cv2.imwrite(thumb_path, first_frame)

    print(f'Generated placeholder: {movement}.mp4 ({DURATION_SEC}s, {TOTAL_FRAMES} frames)')


if __name__ == '__main__':
    t0 = time.time()
    for mv in ['bouncing', 'stepping', 'sliding']:
        write_video(mv)
    elapsed = time.time() - t0
    print(f'\nAll placeholders generated in {elapsed:.1f}s → {OUT_DIR}')
