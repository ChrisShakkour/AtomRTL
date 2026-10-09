"""Minimal chat with the configured model via Ollama, for playing around with the model.

No system prompt, no tools. Conversation history is kept between turns.
Starts the Ollama server if it isn't running, and stops it on exit if it started it.

Usage:
    python misc/chat.py

Type 'exit' or press Ctrl+D to quit.
"""

from langchain_core.messages import HumanMessage

from atomrtl.config import DEFAULT
from atomrtl.llm import make_llm, ollama_server

with ollama_server():
    llm = make_llm()
    history = []

    print(f"Chatting with {DEFAULT.model} (type 'exit' to quit)\n")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt:
            continue

        history.append(HumanMessage(prompt))
        print("model> ", end="", flush=True)
        response = None
        for chunk in llm.stream(history):
            print(chunk.content, end="", flush=True)
            response = chunk if response is None else response + chunk
        print("\n")
        history.append(response)
