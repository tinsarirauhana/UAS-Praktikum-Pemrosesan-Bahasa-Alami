"""
analisis_pipeline.py
Menjalankan seluruh corpus audio code-switching melalui pipeline:
STT (Whisper) → LLM (Gemini) → TTS (Coqui)
Menyimpan log hasil dan menghitung metrik evaluasi (WER, CER, Latency)
"""

import os
import sys
import json
import time
import csv
from pathlib import Path

# Tambahkan root proyek ke path agar bisa import dari app/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import transcribe_text_to_speech

# ============================================================
# KONFIGURASI
# ============================================================

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "data", "corpus", "audio")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "corpus", "output_tts")
LOG_DIR = os.path.join(os.path.dirname(__file__), "log")
LOG_FILE = os.path.join(LOG_DIR, "pipeline_results.json")
CSV_FILE = os.path.join(LOG_DIR, "pipeline_summary.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Reference transcripts untuk WER/CER (berdasarkan ID ujaran)
REFERENCE_TRANSCRIPTS = {
    "audio01": "Aku mau book flight ke Jeddah minggu depan, bisa bantu schedule?",
    "audio02": "Aku butuh travel umrah simple tapi include Madinah visit",
    "audio03": "Can you help aku arrange transport dari Jeddah ke Madinah tomorrow",
    "audio04": "Explain step by step cara apply visa Saudi dengan benar",
    "audio05": "Ya akhi, uridu book flight ila Jeddah al-usbu al qadim. Hal bisa bantu ajida afdhal schedule wa rihlatan mubashirah?",
    "audio06": "Uridu arrange transport min Jeddah ila Madinah ghadan",
    "audio07": "Book flight ke Jeddah lalu lanjut ke Madinah, schedule terbaik kapan",
    "audio08": "Uridu schedule trip min Jeddah ila Makkah bukra sabah",
    "audio09": "Mumkin book transport min Makkah ila Madinah untuk besok",
    "audio10": "Apa perbedaan umrah dan hajj secara detail dalam Islam",
    "audio11": "Kenapa fasting di ramadan itu wajib bagi Muslim",
    "audio12": "Bagaimana proses visa Saudi untuk umrah dari Indonesia sekarang",
    "audio13": "Jelaskan step by step cara booking flight ke Jeddah secara online",
    "audio14": "How to prepare dokumen umrah dari Indonesia dengan benar",
    "audio15": "Tolong buat checklist persiapan umrah termasuk barang wajib dibawa",
    "audio16": "Guide aku cara pilih hotel di Makkah dekat Haram dengan budget terbatas",
    "audio17": "Menurut kamu belajar bahasa Arab itu susah gak untuk pemula",
    "audio18": "I feel overwhelmed dengan persiapan umrah, ada tips sederhana",
    "audio19": "Ahyanan saya bingung mulai dari mana untuk umrah",
    "audio20": "Translate ke English aku mau pergi ke Makkah minggu depan",
}

# ============================================================
# FUNGSI METRIK
# ============================================================

def compute_wer(reference: str, hypothesis: str) -> float:
    """Hitung Word Error Rate (WER)"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if len(ref_words) == 0:
        return 0.0
    
    # Dynamic programming untuk edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)

def compute_cer(reference: str, hypothesis: str) -> float:
    """Hitung Character Error Rate (CER)"""
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))
    
    if len(ref_chars) == 0:
        return 0.0
    
    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    
    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)

def get_utterance_id(filename: str) -> str:
    """Ekstrak utterance ID dari nama file, contoh: 2030_audio01.wav -> audio01"""
    name = Path(filename).stem  # hapus ekstensi
    parts = name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[1:])  # ambil bagian setelah ID mahasiswa
    return name

# ============================================================
# PIPELINE UTAMA
# ============================================================

def run_pipeline(audio_path: str, utterance_id: str) -> dict:
    """Jalankan satu file audio melalui pipeline lengkap"""
    result = {
        "file": os.path.basename(audio_path),
        "utterance_id": utterance_id,
        "stt_transcript": None,
        "llm_response": None,
        "tts_output_path": None,
        "stt_latency_s": None,
        "llm_latency_s": None,
        "tts_latency_s": None,
        "total_latency_s": None,
        "wer": None,
        "cer": None,
        "error": None,
    }

    total_start = time.time()

    # --- STT ---
    try:
        with open(audio_path, "rb") as f:
            file_bytes = f.read()
        
        t0 = time.time()
        transcript = transcribe_speech_to_text(file_bytes, ".wav")
        result["stt_latency_s"] = round(time.time() - t0, 3)
        result["stt_transcript"] = transcript
        print(f"  [STT] {transcript[:80]}...")

        if transcript.startswith("[ERROR]"):
            result["error"] = transcript
            return result
    except Exception as e:
        result["error"] = f"STT Exception: {str(e)}"
        return result

    # --- Hitung WER/CER jika ada reference ---
    ref = REFERENCE_TRANSCRIPTS.get(utterance_id)
    if ref:
        result["wer"] = round(compute_wer(ref, transcript), 4)
        result["cer"] = round(compute_cer(ref, transcript), 4)

    # --- LLM ---
    try:
        t0 = time.time()
        response = generate_response(transcript)
        result["llm_latency_s"] = round(time.time() - t0, 3)
        result["llm_response"] = response
        print(f"  [LLM] {response[:80]}...")

        if response.startswith("[ERROR]"):
            if "429" in response:
                # Retry sekali setelah tunggu
                print("  [LLM] Rate limit hit, tunggu 65 detik...")
                time.sleep(65)
                t0 = time.time()
                response = generate_response(transcript)
                result["llm_latency_s"] = round(time.time() - t0, 3)
                result["llm_response"] = response
                print(f"  [LLM] Retry: {response[:80]}...")

            if response.startswith("[ERROR]"):
                # Masih error setelah retry, skip TTS
                result["error"] = response
                return result
    except Exception as e:
        result["error"] = f"LLM Exception: {str(e)}"
        return result

    # --- TTS ---
    try:
        t0 = time.time()
        tts_path = transcribe_text_to_speech(response)
        result["tts_latency_s"] = round(time.time() - t0, 3)

        if tts_path.startswith("[ERROR]"):
            result["error"] = tts_path
            return result

        # Salin output TTS ke folder output
        import shutil
        out_filename = f"{Path(audio_path).stem}_response.wav"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        shutil.copy2(tts_path, out_path)
        result["tts_output_path"] = out_path
        print(f"  [TTS] Saved: {out_filename}")
    except Exception as e:
        result["error"] = f"TTS Exception: {str(e)}"
        return result

    result["total_latency_s"] = round(time.time() - total_start, 3)
    return result

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("ANALISIS PIPELINE - Code-Switching Speech-to-Speech")
    print("=" * 60)

    # Kumpulkan semua file audio
    audio_files = sorted([
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(".wav")
    ])

    print(f"Total audio ditemukan: {len(audio_files)}")
    print(f"Output TTS: {OUTPUT_DIR}")
    print(f"Log: {LOG_FILE}")
    print("=" * 60)

    # Load progress sebelumnya jika ada (resume mode)
    all_results = []
    processed_files = set()
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            processed_files = set(
                r["file"] for r in all_results if not r.get("error")
            )
            print(f"[RESUME] Melanjutkan dari log sebelumnya...")
            print(f"[RESUME] Sudah berhasil: {len(processed_files)} file")
            print(f"[RESUME] Sisa: {len(audio_files) - len(processed_files)} file")
        except Exception as e:
            print(f"[RESUME] Gagal load log: {e}, mulai dari awal")
            all_results = []

    errors = []

    for i, filename in enumerate(audio_files):
        # Skip jika sudah berhasil diproses sebelumnya
        if filename in processed_files:
            print(f"[{i+1}/{len(audio_files)}] SKIP (sudah diproses): {filename}")
            continue

        audio_path = os.path.join(AUDIO_DIR, filename)
        utterance_id = get_utterance_id(filename)

        print(f"\n[{i+1}/{len(audio_files)}] Memproses: {filename} (utterance: {utterance_id})")

        result = run_pipeline(audio_path, utterance_id)
        all_results.append(result)

        if result["error"]:
            errors.append(filename)
            print(f"  [ERROR] {result['error']}")

        # Simpan log sementara setiap 10 file
        if (i + 1) % 10 == 0:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"  [LOG] Progress saved ({i+1}/{len(audio_files)})")

        # Jeda kecil antar request LLM untuk hindari rate limit
        time.sleep(5)

    # Simpan log final JSON
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Simpan CSV summary
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file", "utterance_id", "stt_transcript", "llm_response",
            "tts_output_path", "stt_latency_s", "llm_latency_s",
            "tts_latency_s", "total_latency_s", "wer", "cer", "error"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # ============================================================
    # RINGKASAN EVALUASI
    # ============================================================
    print("\n" + "=" * 60)
    print("RINGKASAN EVALUASI")
    print("=" * 60)

    successful = [r for r in all_results if not r["error"]]
    failed = [r for r in all_results if r["error"]]

    print(f"Total diproses  : {len(all_results)}")
    print(f"Berhasil        : {len(successful)}")
    print(f"Gagal           : {len(failed)}")

    if successful:
        wer_vals = [r["wer"] for r in successful if r["wer"] is not None]
        cer_vals = [r["cer"] for r in successful if r["cer"] is not None]
        stt_lat  = [r["stt_latency_s"] for r in successful if r["stt_latency_s"]]
        llm_lat  = [r["llm_latency_s"] for r in successful if r["llm_latency_s"]]
        tts_lat  = [r["tts_latency_s"] for r in successful if r["tts_latency_s"]]
        total_lat = [r["total_latency_s"] for r in successful if r["total_latency_s"]]

        if wer_vals:
            print(f"\nSTT Metrics:")
            print(f"  Avg WER : {sum(wer_vals)/len(wer_vals):.4f} ({sum(wer_vals)/len(wer_vals)*100:.1f}%)")
            print(f"  Avg CER : {sum(cer_vals)/len(cer_vals):.4f} ({sum(cer_vals)/len(cer_vals)*100:.1f}%)")

        print(f"\nLatency (rata-rata):")
        if stt_lat:
            print(f"  STT     : {sum(stt_lat)/len(stt_lat):.2f}s")
        if llm_lat:
            print(f"  LLM     : {sum(llm_lat)/len(llm_lat):.2f}s")
        if tts_lat:
            print(f"  TTS     : {sum(tts_lat)/len(tts_lat):.2f}s")
        if total_lat:
            print(f"  Total   : {sum(total_lat)/len(total_lat):.2f}s")

    if failed:
        print(f"\nFile yang gagal diproses:")
        for r in failed:
            print(f"  - {r['file']}: {r['error'][:60]}")

    print(f"\nLog lengkap: {LOG_FILE}")
    print(f"CSV summary: {CSV_FILE}")
    print("=" * 60)
    print("SELESAI!")

if __name__ == "__main__":
    main()