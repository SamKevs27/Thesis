"""
Pre-compute pose cache for reference videos.

Reads assets/references/{movement}.mp4, extracts joint angles via
app.py's extract_angles_from_video pipeline, and writes a JSON cache
to assets/reference_cache/{movement}.json.

Run OUTSIDE Flask (standalone script):
    python scripts/precompute_references.py

Re-run any time a reference video is replaced.
"""

import os
import sys
import json
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Resolve project root so we can import app-level helpers
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

REFERENCES_DIR = os.path.join(PROJECT_ROOT, 'assets', 'references')
CACHE_DIR      = os.path.join(PROJECT_ROOT, 'assets', 'reference_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

MOVEMENTS = ['bouncing', 'stepping', 'sliding']
MIN_FRAMES = 30     # below this → too_short


# ---------------------------------------------------------------------------
# Import extraction helpers from app
# ---------------------------------------------------------------------------
def _import_helpers():
    """
    Import extract_angles_from_video (and optionally smooth_pose_data)
    from app.py without triggering the Flask server.

    We monkey-patch mediapipe's solution stub before import to avoid
    the AttributeError that can occur in headless environments.
    """
    try:
        from app import extract_angles_from_video
        # Try optional smoother — gracefully absent
        try:
            from app import smooth_pose_data
        except ImportError:
            smooth_pose_data = None
        return extract_angles_from_video, smooth_pose_data
    except Exception as e:
        print(f'[ERROR] Failed to import from app.py: {e}', file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Cache one movement
# ---------------------------------------------------------------------------

def precompute_one(movement: str,
                   extract_fn,
                   smooth_fn) -> dict:
    """
    Extract pose from the reference video for *movement*.

    Returns a dict ready to be JSON-serialised, or raises on fatal error.
    """
    video_path = os.path.join(REFERENCES_DIR, f'{movement}.mp4')

    if not os.path.exists(video_path):
        return {
            'movement_type':   movement,
            'video_path':      video_path,
            'video_filename':  f'{movement}.mp4',
            'extracted_at':    datetime.now().isoformat(),
            'extraction_status': 'missing',
            'fps':             0.0,
            'frame_count':     0,
            'duration_sec':    0.0,
            'joint_angles':    [],
            'timestamps':      [],
        }

    t0 = time.time()
    try:
        angles, timestamps, landmarks, error = extract_fn(video_path)
    except Exception as exc:
        return {
            'movement_type':   movement,
            'video_path':      video_path,
            'video_filename':  f'{movement}.mp4',
            'extracted_at':    datetime.now().isoformat(),
            'extraction_status': 'error',
            'error_message':   str(exc),
            'fps':             0.0,
            'frame_count':     0,
            'duration_sec':    0.0,
            'joint_angles':    [],
            'timestamps':      [],
        }

    if error:
        return {
            'movement_type':   movement,
            'video_path':      video_path,
            'video_filename':  f'{movement}.mp4',
            'extracted_at':    datetime.now().isoformat(),
            'extraction_status': 'error',
            'error_message':   error,
            'fps':             0.0,
            'frame_count':     0,
            'duration_sec':    0.0,
            'joint_angles':    [],
            'timestamps':      [],
            'landmarks':       [],
        }

    import numpy as np
    angles     = np.array(angles,     dtype=float)
    timestamps = np.array(timestamps, dtype=float)
    n_frames   = len(angles)

    # Optional smoothing pass
    if smooth_fn is not None and n_frames > 5:
        try:
            angles = smooth_fn(angles)
        except Exception:
            pass

    elapsed = time.time() - t0
    fps_est = float(n_frames / timestamps[-1]) if timestamps[-1] > 0 else 30.0

    if n_frames < MIN_FRAMES:
        status = 'too_short'
    elif n_frames < 60:
        status = 'degraded'
    else:
        status = 'ok'

    return {
        'movement_type':   movement,
        'video_path':      os.path.relpath(video_path, PROJECT_ROOT),
        'video_filename':  f'{movement}.mp4',
        'extracted_at':    datetime.now().isoformat(),
        'extraction_status': status,
        'fps':             round(fps_est, 2),
        'frame_count':     n_frames,
        'duration_sec':    round(float(timestamps[-1]) if n_frames > 0 else 0.0, 3),
        'joint_angles':    angles.tolist(),
        'timestamps':      timestamps.tolist(),
        'landmarks':       landmarks if landmarks is not None else [],
        '_elapsed_sec':    round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print('Pre-computing reference pose cache …\n')

    extract_fn, smooth_fn = _import_helpers()

    ok_count       = 0
    degraded_count = 0
    skip_count     = 0

    for movement in MOVEMENTS:
        print(f'  Processing {movement} …', end='', flush=True)
        result = precompute_one(movement, extract_fn, smooth_fn)

        status    = result['extraction_status']
        n_frames  = result['frame_count']
        elapsed   = result.get('_elapsed_sec', 0)

        cache_path = os.path.join(CACHE_DIR, f'{movement}.json')

        if status == 'missing':
            print(f' SKIPPED (file not found: assets/references/{movement}.mp4)')
            skip_count += 1
            continue

        # Save cache regardless of status (loader will skip 'too_short')
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)

        if status in ('ok', 'degraded'):
            marker = '✓' if status == 'ok' else '~'
            note   = '' if status == 'ok' else ' (degraded — few landmarks detected)'
            print(f' {marker} {movement}: {n_frames} frames extracted in {elapsed}s{note}')
            if status == 'degraded':
                degraded_count += 1
            else:
                ok_count += 1
        elif status == 'too_short':
            print(f' ✗ {movement}: only {n_frames} frames detected (video too short or pose unclear)')
            skip_count += 1
        else:
            msg = result.get('error_message', 'unknown error')
            print(f' ✗ {movement}: error — {msg}')
            skip_count += 1

    # Summary
    total_done = ok_count + degraded_count
    total      = len(MOVEMENTS)
    print()
    if skip_count == 0:
        print(f'Cached {total_done}/{total} references successfully')
    else:
        suffix = f' ({degraded_count} degraded)' if degraded_count else ''
        print(f'Cached {total_done}/{total} references{suffix} ({skip_count} skipped/failed)')
    print(f'Cache written to: {CACHE_DIR}')


if __name__ == '__main__':
    main()
