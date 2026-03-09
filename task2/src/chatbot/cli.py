import os
import itertools
import threading
import time

from dotenv import load_dotenv
import requests

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000") + "/chat"
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "240"))


def _spinner(stop_event: threading.Event) -> None:
    for ch in itertools.cycle(["|", "/", "-", "\\"]):
        if stop_event.is_set():
            break
        print(f"\rGenerating recipe... {ch}", end="", flush=True)
        time.sleep(0.1)
    print("\r" + " " * 40 + "\r", end="", flush=True)


def main():
    print("Recipe Chatbot. Type ingredients or 'quit'.")
    print(f"API: {API_URL}")
    while True:
        text = input("> ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        stop = threading.Event()
        t = threading.Thread(target=_spinner, args=(stop,), daemon=True)
        t.start()
        try:
            resp = requests.post(API_URL, json={"ingredients": text}, timeout=API_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        finally:
            stop.set()
            t.join()
        print(data["recipe"])
        print(f"Notes: {data['notes']}")


if __name__ == "__main__":
    main()
