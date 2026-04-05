# Blender Socket Server — install this as a Blender add-on
# Listens on localhost:7777 for Python code strings and executes them in Blender.

bl_info = {
    "name": "Claude Socket Server",
    "description": "Receives and executes Python code from the Claude terminal",
    "author": "AI Portfolio",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "category": "AI",
}

import bpy
import socket
import threading

PORT = 7777
_pending_code = None
_server_thread = None
_server_socket = None


def run_server():
    global _server_socket
    _server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _server_socket.bind(("localhost", PORT))
    _server_socket.listen(5)
    _server_socket.settimeout(1.0)
    print(f"[Claude] Socket server listening on port {PORT}")
    while True:
        try:
            conn, _ = _server_socket.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        chunks = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        conn.close()
        global _pending_code
        _pending_code = b"".join(chunks).decode("utf-8")


def poll_and_execute():
    global _pending_code
    if _pending_code is not None:
        code = _pending_code
        _pending_code = None
        try:
            exec(compile(code, "<claude>", "exec"), {"bpy": bpy})
            print("[Claude] Code executed successfully")
        except Exception as e:
            print(f"[Claude] Execution error: {e}")
        # Redraw
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    return 0.2  # keep polling every 200ms


def register():
    global _server_thread
    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()
    bpy.app.timers.register(poll_and_execute, persistent=True)
    print(f"[Claude] Server started on port {PORT}")


def unregister():
    global _server_socket
    bpy.app.timers.unregister(poll_and_execute)
    if _server_socket:
        _server_socket.close()
