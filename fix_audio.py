"""
fix_audio.py
Convert hanya file-file audio yang bermasalah ke format standar.
"""

import os
import subprocess

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "data", "corpus", "audio")

PROBLEMATIC_FILES = [
    "2362_audio20.wav", "2362_audio17.wav", "2362_audio15.wav",
    "2362_audio07.wav", "2362_audio06.wav", "2362_audio05.wav",
    "2362_audio04.wav", "2362_audio03.wav", "2362_audio02.wav",
    "2362_audio01.wav", "2362_audio11.wav",
    "2347_audio7.wav", "2347_audio6.wav", "2347_audio5.wav",
    "2347_audio4.wav", "2347_audio3.wav", "2347_audio2.wav",
    "2347_audio1.wav", "2347_audio17.wav", "2347_audio14.wav",
    "2347_audio11.wav",
    "2341_audio10.wav", "2341_audio09.wav", "2341_audio08.wav",
    "2341_audio07.wav", "2341_audio06.wav", "2341_audio05.wav",
    "2341_audio04.wav", "2341_audio03.wav", "2341_audio02.wav",
    "2341_audio01.wav",
    "2339_audio19.wav", "2339_audio14.wav", "2339_audio11.wav",
    "2339_audio07.wav", "2339_audio06.wav", "2339_audio05.wav",
    "2339_audio04.wav", "2339_audio03.wav", "2339_audio02.wav",
    "2339_audio01.wav",
    "2338_audio17.wav", "2338_audio15.wav", "2338_audio13.wav",
    "2338_audio12.wav", "2338_audio06.wav", "2338_audio05.wav",
    "2338_audio04.wav", "2338_audio03.wav", "2338_audio02.wav",
    "2338_audio01.wav",
]

def convert(input_path, output_path):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] {e.stderr.decode()[:100]}")
        return False

def main():
    print(f"Fix {len(PROBLEMATIC_FILES)} file bermasalah...\n")
    success, failed = 0, []

    for filename in PROBLEMATIC_FILES:
        input_path = os.path.join(AUDIO_DIR, filename)
        tmp_path = input_path + ".tmp.wav"

        if not os.path.exists(input_path):
            print(f"SKIP (tidak ditemukan): {filename}")
            continue

        print(f"Converting: {filename} ... ", end="")
        if convert(input_path, tmp_path):
            os.replace(tmp_path, input_path)
            print("OK")
            success += 1
        else:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print("GAGAL")
            failed.append(filename)

    print(f"\nSelesai! Berhasil: {success}/{len(PROBLEMATIC_FILES)}")
    if failed:
        print("Gagal:")
        for f in failed:
            print(f"  - {f}")
    print("\nSekarang jalankan: python analisis_pipeline.py")

if __name__ == "__main__":
    main()
