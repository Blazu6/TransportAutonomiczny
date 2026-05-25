import subprocess
from pathlib import Path


TS_FILE = Path("output/Prototype_hevc.ts")
DESTINATION_IP = "127.0.0.1"
DESTINATION_PORT = 5005


def stream_ts_file(ts_file: Path, ip: str, port: int) -> None:
    if not ts_file.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {ts_file}")

    command = [
        "ffmpeg",
        "-re",
        "-i", str(ts_file),
        "-c", "copy",
        "-f", "mpegts",
        f"udp://{ip}:{port}"
    ]

    subprocess.run(command, check=True)


if __name__ == "__main__":
    stream_ts_file(TS_FILE, DESTINATION_IP, DESTINATION_PORT)