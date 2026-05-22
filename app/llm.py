import os
import time
from google import genai
from google.genai import types
from pydantic import TypeAdapter
from dotenv import load_dotenv

load_dotenv()

MODEL = "models/gemma-4-26b-a4b-it"

# Kumpulkan semua API key dari .env (auto-rotate jika rate limit)
def _load_api_keys():
    keys = []
    # Key utama
    k = os.getenv("GEMINI_API_KEY")
    if k:
        keys.append(k)
    # Key tambahan GEMINI_API_KEY_1 dst
    i = 1
    while True:
        k = os.getenv(f"GEMINI_API_KEY_{i}")
        if not k:
            break
        keys.append(k)
        i += 1
    return keys

API_KEYS = _load_api_keys()
current_key_index = 0
print(f"[LLM] Loaded {len(API_KEYS)} API key(s)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

system_instruction = """
You are a responsive, intelligent, and fluent virtual assistant who communicates in Indonesian.
Your task is to provide clear, concise, and informative answers in response to user queries or statements spoken through voice.

Your answers must:
- Be written in polite and easily understandable Indonesian.
- Be short and to the point (maximum 2–3 sentences).
- Avoid repeating the user's question; respond directly with the answer.

Example tone:
User: Cuaca hari ini gimana?
Assistant: Hari ini cuacanya cerah di sebagian besar wilayah, dengan suhu sekitar 30 derajat.

User: Kamu tahu siapa presiden Indonesia?
Assistant: Presiden Indonesia saat ini adalah Prabowo Subianto.

If you're unsure about an answer, be honest and say that you don't know.
"""

chat_config = types.GenerateContentConfig(system_instruction=system_instruction)
history_adapter = TypeAdapter(list[types.Content])

def _make_client(key_index: int):
    return genai.Client(api_key=API_KEYS[key_index])

def _make_chat(client):
    return client.chats.create(model=MODEL, config=chat_config)

# Inisialisasi client dan chat dengan key pertama
client = _make_client(current_key_index)
chat = _make_chat(client)

def generate_response(prompt: str) -> str:
    global current_key_index, client, chat

    max_retries = len(API_KEYS)
    for attempt in range(max_retries):
        try:
            response = chat.send_message(prompt)
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print(f"  [LLM] Key [{current_key_index}] rate limit! Ganti key berikutnya...")
                current_key_index = (current_key_index + 1) % len(API_KEYS)
                client = _make_client(current_key_index)
                chat = _make_chat(client)
                print(f"  [LLM] Pakai key [{current_key_index}], tunggu 5 detik...")
                time.sleep(5)
            else:
                return f"[ERROR] {err}"

    return "[ERROR] Semua API key habis quota atau gagal."