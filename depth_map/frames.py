import subprocess
from pathlib import Path


def extract_frames(video_path, out_dir, fps=None, scale_width=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.png")
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    filters = []
    if fps:
        filters.append(f"fps={fps}")
    if scale_width:
        # -2 keeps height a multiple of 2 (required by most codecs/filters)
        # while preserving aspect ratio.
        filters.append(f"scale={scale_width}:-2")
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += [pattern]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return sorted(out_dir.glob("frame_*.png"))
