"""
evaluate_pipeline.py
Bersihkan duplikat dari pipeline_results.json dan hitung metrik evaluasi.
Logic: per file, ambil hasil TERBAIK (yang berhasil / tidak error).
Kalau semua run error, ambil yang terakhir.
"""

import json
import os
import csv

LOG_FILE = os.path.join(os.path.dirname(__file__), "log", "pipeline_results.json")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "log", "pipeline_results_clean.json")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "log", "pipeline_summary_clean.csv")

def main():
    # Load log
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total baris di log (dengan duplikat): {len(data)}")

    # Deduplicate: per file, prioritaskan yang berhasil (tidak error)
    best = {}
    for r in data:
        fname = r["file"]
        if fname not in best:
            best[fname] = r
        else:
            # Kalau yang baru berhasil (tidak error), pakai yang baru
            if not r.get("error") and best[fname].get("error"):
                best[fname] = r
            # Kalau keduanya berhasil, pakai yang terbaru (tetap yang sudah ada)
            # Kalau keduanya error, pakai yang terbaru
            elif r.get("error") and best[fname].get("error"):
                best[fname] = r

    clean = list(best.values())
    print(f"Total file unik: {len(clean)}")

    sukses = [r for r in clean if not r.get("error")]
    gagal = [r for r in clean if r.get("error")]
    print(f"Berhasil: {len(sukses)}")
    print(f"Gagal   : {len(gagal)}")

    # Simpan clean JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    # Simpan clean CSV
    fieldnames = [
        "file", "utterance_id", "stt_transcript", "llm_response",
        "tts_output_path", "stt_latency_s", "llm_latency_s",
        "tts_latency_s", "total_latency_s", "wer", "cer", "error"
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean)

    # ============================================================
    # HITUNG METRIK
    # ============================================================
    print("\n" + "=" * 60)
    print("HASIL EVALUASI PIPELINE")
    print("=" * 60)

    print(f"\nOverview:")
    print(f"  Total audio unik     : {len(clean)}")
    print(f"  Berhasil full pipeline: {len(sukses)} ({len(sukses)/len(clean)*100:.1f}%)")
    print(f"  Gagal                : {len(gagal)} ({len(gagal)/len(clean)*100:.1f}%)")

    # WER & CER
    wer_vals = [r["wer"] for r in sukses if r.get("wer") is not None]
    cer_vals = [r["cer"] for r in sukses if r.get("cer") is not None]

    if wer_vals:
        print(f"\nSTT Metrics (dari {len(wer_vals)} audio dengan reference):")
        print(f"  Avg WER : {sum(wer_vals)/len(wer_vals):.4f} ({sum(wer_vals)/len(wer_vals)*100:.1f}%)")
        print(f"  Min WER : {min(wer_vals):.4f} ({min(wer_vals)*100:.1f}%)")
        print(f"  Max WER : {max(wer_vals):.4f} ({max(wer_vals)*100:.1f}%)")
        print(f"  Avg CER : {sum(cer_vals)/len(cer_vals):.4f} ({sum(cer_vals)/len(cer_vals)*100:.1f}%)")
        print(f"  Min CER : {min(cer_vals):.4f} ({min(cer_vals)*100:.1f}%)")
        print(f"  Max CER : {max(cer_vals):.4f} ({max(cer_vals)*100:.1f}%)")

    # Latency
    stt_lat  = [r["stt_latency_s"] for r in sukses if r.get("stt_latency_s")]
    llm_lat  = [r["llm_latency_s"] for r in sukses if r.get("llm_latency_s")]
    tts_lat  = [r["tts_latency_s"] for r in sukses if r.get("tts_latency_s")]
    total_lat = [r["total_latency_s"] for r in sukses if r.get("total_latency_s")]

    print(f"\nLatency (rata-rata, dari {len(sukses)} audio berhasil):")
    if stt_lat:
        print(f"  STT   : {sum(stt_lat)/len(stt_lat):.2f}s (min: {min(stt_lat):.2f}s, max: {max(stt_lat):.2f}s)")
    if llm_lat:
        print(f"  LLM   : {sum(llm_lat)/len(llm_lat):.2f}s (min: {min(llm_lat):.2f}s, max: {max(llm_lat):.2f}s)")
    if tts_lat:
        print(f"  TTS   : {sum(tts_lat)/len(tts_lat):.2f}s (min: {min(tts_lat):.2f}s, max: {max(tts_lat):.2f}s)")
    if total_lat:
        print(f"  Total : {sum(total_lat)/len(total_lat):.2f}s (min: {min(total_lat):.2f}s, max: {max(total_lat):.2f}s)")

    # Analisis error
    if gagal:
        error_types = {}
        for r in gagal:
            err = r.get("error", "Unknown")
            if "503" in err:
                key = "503 UNAVAILABLE (server overload)"
            elif "429" in err:
                key = "429 RATE_LIMIT"
            elif "403" in err:
                key = "403 PERMISSION_DENIED"
            elif "STT" in err or "Whisper" in err:
                key = "STT Error (audio bermasalah)"
            elif "TTS" in err:
                key = "TTS Error"
            else:
                key = "Error lainnya"
            error_types[key] = error_types.get(key, 0) + 1

        print(f"\nBreakdown Error ({len(gagal)} file):")
        for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {etype}: {count} file")

    # Per utterance ID analysis
    print(f"\nHasil per Utterance ID:")
    utterance_stats = {}
    for r in clean:
        uid = r.get("utterance_id", "unknown")
        if uid not in utterance_stats:
            utterance_stats[uid] = {"total": 0, "sukses": 0}
        utterance_stats[uid]["total"] += 1
        if not r.get("error"):
            utterance_stats[uid]["sukses"] += 1

    for uid in sorted(utterance_stats.keys()):
        s = utterance_stats[uid]
        print(f"  {uid}: {s['sukses']}/{s['total']} berhasil")

    print(f"\nFile clean tersimpan:")
    print(f"  JSON: {OUTPUT_JSON}")
    print(f"  CSV : {OUTPUT_CSV}")
    print("=" * 60)

if __name__ == "__main__":
    main()
