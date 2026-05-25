import subprocess
from pathlib import Path


INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")


def get_mp4_from_input() -> Path:
    mp4_files = list(INPUT_DIR.glob("*.mp4"))

    if not mp4_files:
        raise FileNotFoundError("Brak pliku MP4 w folderze input")

    return mp4_files[0]


def convert_mp4_to_mpegts(
    input_path: Path,
    output_path: Path,
    crf: int = 28,
    width: int = 1280
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vf", f"scale={width}:-2",
        "-c:v", "libx265",
        "-crf", str(crf),
        "-f", "mpegts",
        str(output_path)
    ]

    subprocess.run(command, check=True)


if __name__ == "__main__":
    input_file = get_mp4_from_input()
    output_file = OUTPUT_DIR / f"{input_file.stem}_hevc.ts"

    print(f"Plik wejściowy: {input_file}")
    print(f"Plik wyjściowy: {output_file}")

    convert_mp4_to_mpegts(
        input_path=input_file,
        output_path=output_file,
        crf=28,
        width=1280
    )

    print("Konwersja do MPEG-TS zakończona.")
