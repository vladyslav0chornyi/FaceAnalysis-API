import os
import subprocess

VIDEO_DIR = "videos"
N_STUBS = 30

if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

for i in range(1, N_STUBS + 1):
    stub_path = os.path.join(VIDEO_DIR, f"phrase_{i:02d}.mp4")
    if not os.path.exists(stub_path):
        # Створити 2-секундне чорне відео 640x360 з тишею
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=640x360:d=2",
            "-f", "lavfi", "-i", "anullsrc",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            stub_path
        ]
        subprocess.run(cmd)
        print(f"Created stub {stub_path}")
    else:
        print(f"Already exists: {stub_path}")