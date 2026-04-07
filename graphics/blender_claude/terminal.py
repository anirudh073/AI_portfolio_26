#!/usr/bin/env python3
# Claude → Blender terminal
# Usage: python3 terminal.py
# Type your prompt, press Enter. Claude generates bpy code, it runs in Blender.

import subprocess
import socket
import re
import sys

BLENDER_HOST = "localhost"
BLENDER_PORT = 7777

SYSTEM_PROMPT = """\
You are an expert Blender Python (bpy) programmer for Blender 4.x.
Respond with a brief explanation, then a single ```python code block.
Rules:
- Wrap all code in execute_claude_command() and call it at the end
- Do NOT import bpy — it is already available
- Use correct Blender 4.x API
- Handle edge cases (check objects exist before accessing)
"""


def extract_code(text):
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def ask_claude(full_prompt):
    result = subprocess.run(
        ["claude", "-p", "--output-format", "text"],
        input=full_prompt,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "claude CLI error")
    return result.stdout.strip()


def send_to_blender(code):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5)
        s.connect((BLENDER_HOST, BLENDER_PORT))
        s.sendall(code.encode("utf-8"))


def build_prompt(user_message, history):
    parts = [SYSTEM_PROMPT, ""]
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        parts.append(f"{role}: {turn['content']}")
    parts.append(f"User: {user_message}")
    return "\n\n".join(parts)


def read_prompt():
    """Collect one or more lines; blank line submits. Handles long pastes."""
    print("You (blank line to send):")
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return None
        if line == "":
            break
        lines.append(line)
    return " ".join(lines).strip()


def main():
    print("=" * 50)
    print("  Claude → Blender Terminal")
    print("  Commands: 'clear' to reset history, 'quit' to exit")
    print("  Paste your prompt, then press Enter on a blank line to send")
    print("=" * 50)

    # Check Blender connection
    try:
        send_to_blender("# ping")
        print("  Blender: connected ✓")
    except Exception:
        print("  Blender: NOT connected — install blender_socket_server.py add-on first")

    print()

    history = []

    while True:
        prompt = read_prompt()
        if prompt is None:
            print("\nBye.")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            break
        if prompt.lower() == "clear":
            history = []
            print("History cleared.\n")
            continue

        # Call Claude
        print("Claude: thinking...", end="\r")
        try:
            full_prompt = build_prompt(prompt, history)
            response = ask_claude(full_prompt)
        except Exception as e:
            print(f"Claude: error — {e}\n")
            continue

        # Print response (strip code block for cleaner output)
        display = re.sub(r"```python.*?```", "[code block]", response, flags=re.DOTALL).strip()
        print(f"Claude: {display}\n")

        # Extract and send code
        code = extract_code(response)
        if code:
            try:
                send_to_blender(code)
                print("  → Code sent to Blender ✓\n")
            except Exception as e:
                print(f"  → Could not reach Blender: {e}\n")
                print("  Code:\n")
                print(code)
                print()
        else:
            print("  (no code generated)\n")

        # Update history
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
