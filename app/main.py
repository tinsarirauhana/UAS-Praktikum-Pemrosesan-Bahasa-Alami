import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import transcribe_text_to_speech

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Voice Chatbot API is running."}

@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):
    """
    Endpoint utama: terima audio -> STT -> LLM -> TTS -> kembalikan audio
    """
    file_bytes = await file.read()
    file_ext = os.path.splitext(file.filename)[-1] or ".wav"

    # STT
    transcript = transcribe_speech_to_text(file_bytes, file_ext)
    print(f"[STT] Transkripsi: {transcript}")
    if transcript.startswith("[ERROR]"):
        raise HTTPException(status_code=500, detail=transcript)

    # LLM
    response_text = generate_response(transcript)
    print(f"[LLM] Respons: {response_text}")
    if response_text.startswith("[ERROR]"):
        raise HTTPException(status_code=500, detail=response_text)

    # TTS
    audio_path = transcribe_text_to_speech(response_text)
    print(f"[TTS] Audio: {audio_path}")
    if audio_path.startswith("[ERROR]"):
        raise HTTPException(status_code=500, detail=audio_path)

    # Return audio + teks di headers
    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename="response.wav",
        headers={
            "X-Transcript": transcript.replace("\n", " "),
            "X-LLM-Response": response_text.replace("\n", " "),
            "Access-Control-Expose-Headers": "X-Transcript, X-LLM-Response"
        }
    )