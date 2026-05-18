"""
Reference pose cache loader.

Loads pre-computed joint-angle JSON files from assets/reference_cache/
into memory at Flask startup.  Provides thread-safe read access for the
upload-student grading pipeline.

Usage (in app.py):
    from reference_loader import (load_all_references, get_reference,
                                  has_reference, list_available_references,
                                  get_all_references_meta)

    # called once at startup (inside app context is not required)
    load_all_references()
"""

import json
import os
import numpy as np
from threading import Lock

CACHE_DIR = 'assets/reference_cache'

_references: dict = {}
_lock = Lock()


def load_all_references() -> None:
    """Load all cached references into memory at startup."""
    global _references
    with _lock:
        _references.clear()

        if not os.path.exists(CACHE_DIR):
            print(f'[reference] Cache dir not found: {CACHE_DIR}')
            return

        loaded = []
        for fname in sorted(os.listdir(CACHE_DIR)):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(CACHE_DIR, fname)
            try:
                with open(path) as f:
                    data = json.load(f)

                movement = data['movement_type']
                status   = data.get('extraction_status', 'ok')

                if status == 'missing':
                    print(f'[reference] Skipping {movement}: video file was missing at cache time')
                    continue
                if status == 'too_short':
                    print(f'[reference] Skipping {movement}: too short ({data.get("frame_count", 0)} frames)')
                    continue
                if status == 'error':
                    print(f'[reference] Skipping {movement}: extraction error — {data.get("error_message", "")}')
                    continue

                joint_angles = np.array(data['joint_angles'], dtype=float)
                timestamps   = np.array(data['timestamps'],   dtype=float)

                if joint_angles.ndim != 2 or joint_angles.shape[1] != 13:
                    print(f'[reference] Skipping {movement}: unexpected shape {joint_angles.shape}')
                    continue

                _references[movement] = {
                    'joint_angles':      joint_angles,
                    'timestamps':        timestamps,
                    'landmarks':         data.get('landmarks'),   # None for old caches
                    'fps':               float(data['fps']),
                    'duration_sec':      float(data['duration_sec']),
                    'video_path':        data['video_path'],
                    'video_filename':    data.get('video_filename',
                                                  os.path.basename(data['video_path'])),
                    'frame_count':       int(data['frame_count']),
                    'extraction_status': status,
                    'extracted_at':      data.get('extracted_at', ''),
                }
                loaded.append(movement)

            except Exception as e:
                print(f'[reference] Failed to load {fname}: {e}')

    print(f'[reference] Loaded {len(_references)} references: {list(_references.keys())}')


def get_reference(movement_type: str) -> dict | None:
    """Return the in-memory reference for *movement_type*, or None."""
    return _references.get(movement_type)


def has_reference(movement_type: str) -> bool:
    """True if a valid (non-degraded or degraded but usable) reference exists."""
    return movement_type in _references


def list_available_references() -> list[str]:
    """Return sorted list of movement keys with loaded references."""
    return sorted(_references.keys())


def get_all_references_meta() -> dict:
    """Return metadata for all references (frontend-safe, no numpy arrays)."""
    result = {}
    for movement, ref in _references.items():
        result[movement] = {
            'video_filename':    ref['video_filename'],
            'duration_sec':      ref['duration_sec'],
            'frame_count':       ref['frame_count'],
            'fps':               ref['fps'],
            'extraction_status': ref['extraction_status'],
            'extracted_at':      ref['extracted_at'],
        }
    return result
