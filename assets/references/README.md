# Reference Videos

This folder holds the pre-recorded reference videos for each movement type.

## Expected files

| File | Movement |
|------|----------|
| `bouncing.mp4` | Bouncing |
| `stepping.mp4` | Stepping |
| `sliding.mp4` | Sliding |

Thumbnail stills (`.thumb.jpg`) are generated alongside each video by
`scripts/generate_placeholders.py` and are served to the frontend.

## Workflow

1. **Add / replace a video** — drop the MP4 here with the exact filename above.
2. **Re-run the precompute script** to rebuild the JSON cache:
   ```
   python scripts/precompute_references.py
   ```
3. **Restart Flask** — the cache is loaded into memory at startup by
   `reference_loader.py`.

## Generating placeholder videos (for testing)

If you don't have real reference videos yet, generate animated stick-figure
placeholders:
```
python scripts/generate_placeholders.py
```
MediaPipe detection on these is unreliable (extraction_status = "degraded"),
but the full pipeline can still be exercised end-to-end.

## Notes

- Videos must be MP4, portrait orientation recommended (720 × 1280).
- The precompute script requires at least 30 detected frames; videos with
  fewer detected frames are cached as `too_short` and skipped at load time.
- MP4 files and thumbnails are git-ignored (see `.gitignore`); commit the
  JSON cache files in `assets/reference_cache/` instead if you want
  repeatable builds without re-running the precompute step.
