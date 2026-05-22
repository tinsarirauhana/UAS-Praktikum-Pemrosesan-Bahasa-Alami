import os
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile

def voice_chat(audio):
    if audio is None:
        return None, "Tidak ada audio yang direkam.", ""

    sr, audio_data = audio

    # Simpan sebagai .wav
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
        audio_path = tmpfile.name

    # Kirim ke endpoint FastAPI
    try:
        with open(audio_path, "rb") as f:
            files = {"file": ("voice.wav", f, "audio/wav")}
            response = requests.post("http://localhost:8000/voice-chat", files=files, timeout=180)
    except requests.exceptions.ConnectionError:
        return None, "[ERROR] Backend tidak berjalan. Jalankan uvicorn dulu!", ""
    except requests.exceptions.Timeout:
        return None, "[ERROR] Request timeout. Coba lagi.", ""

    if response.status_code == 200:
        # Ambil teks dari headers
        transcript = response.headers.get("X-Transcript", "-")
        llm_response = response.headers.get("X-LLM-Response", "-")

        # Simpan file audio respons
        output_audio_path = os.path.join(tempfile.gettempdir(), "tts_output.wav")
        with open(output_audio_path, "wb") as f:
            f.write(response.content)

        return output_audio_path, transcript, llm_response
    else:
        detail = response.json().get("detail", "Unknown error")
        return None, f"[ERROR] {detail}", ""

# UI Gradio
with gr.Blocks(title="Voice Chatbot - Code-Switching") as demo:
    gr.Markdown("# 🎙️ Voice Chatbot")
    gr.Markdown("Berbicara langsung ke mikrofon (Bahasa Indonesia / English / العربية) dan dapatkan jawaban suara dari asisten AI.")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources="microphone",
                type="numpy",
                label="🎤 Rekam Pertanyaan Anda"
            )
            submit_btn = gr.Button("🔁 Submit", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(
                type="filepath",
                label="🔊 Balasan dari Asisten"
            )

    with gr.Row():
        with gr.Column():
            transcript_box = gr.Textbox(
                label="📝 Transkripsi STT (hasil Whisper)",
                placeholder="Hasil transkripsi suara kamu akan muncul di sini...",
                interactive=False,
                lines=3
            )
        with gr.Column():
            llm_box = gr.Textbox(
                label="🤖 Respons LLM (Gemini)",
                placeholder="Respons teks dari Gemini akan muncul di sini...",
                interactive=False,
                lines=3
            )

    submit_btn.click(
        fn=voice_chat,
        inputs=audio_input,
        outputs=[audio_output, transcript_box, llm_box]
    )

    gr.Markdown("---")
    gr.Markdown("**Pipeline:** 🎤 Audio → 📝 Whisper STT → 🤖 Gemini LLM → 🔊 Coqui TTS")

demo.launch()