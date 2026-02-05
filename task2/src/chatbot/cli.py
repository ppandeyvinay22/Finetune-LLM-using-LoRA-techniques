import os

from dotenv import load_dotenv
import requests

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000") + "/chat"
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "240"))


def main():
    print("Recipe Chatbot. Type ingredients or 'quit'.")
    print(f"API: {API_URL}")
    while True:
        text = input("> ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        resp = requests.post(API_URL, json={"ingredients": text}, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        print(f"Recipe: {data['recipe']}")
        print(f"Notes: {data['notes']}")


if __name__ == "__main__":
    main()
