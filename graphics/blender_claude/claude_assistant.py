# Claude AI Assistant for Blender
# Uses the Claude Code CLI (`claude -p`) — no API key needed.
# Requires: Claude Code installed and authenticated.

bl_info = {
    "name": "Claude AI Assistant",
    "description": "Control Blender with natural language using Claude Code CLI",
    "author": "AI Portfolio",
    "version": (1, 1),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Claude",
    "category": "AI",
}

import bpy
import json
import subprocess
import threading
import shutil
import re
import os
from bpy.props import StringProperty, BoolProperty
from bpy.types import Panel, Operator, PropertyGroup

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_conversation_history = []   # list of {"role": ..., "content": ...}
_is_loading = False
_pending_result = None       # (response_text, code_text, error) set by thread


SYSTEM_PROMPT = """\
You are an expert Blender Python (bpy) programmer for Blender 4.x.
When the user asks you to do something in Blender, respond with:
1. A brief explanation (1-2 sentences)
2. A single ```python code block containing the implementation

Code rules:
- Wrap all logic in a function called execute_claude_command() and call it at the end
- Do NOT import bpy — it is already available
- Use correct Blender 4.x API (e.g. mat.diffuse_color for simple materials)
- Handle edge cases (check if objects exist before accessing them)
- Add short inline comments

Example:
I'll add a red sphere at the origin.

```python
def execute_claude_command():
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "ClaudeSphere"
    mat = bpy.data.materials.new("Red")
    mat.diffuse_color = (1.0, 0.0, 0.0, 1.0)
    obj.data.materials.append(mat)

execute_claude_command()
```
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_claude_cli():
    """Return path to the claude CLI, or None if not found."""
    # Check common locations
    for candidate in [
        shutil.which("claude"),
        os.path.expanduser("~/.local/bin/claude"),
        "/usr/local/bin/claude",
    ]:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def get_scene_context():
    """Compact JSON summary of the current scene."""
    objects = []
    for obj in list(bpy.context.scene.objects)[:20]:
        objects.append({
            "name": obj.name,
            "type": obj.type,
            "location": [round(v, 3) for v in obj.location],
        })
    render = bpy.context.scene.render
    return json.dumps({
        "objects": objects,
        "render_engine": render.engine,
        "resolution": [render.resolution_x, render.resolution_y],
        "active_object": (bpy.context.active_object.name
                          if bpy.context.active_object else None),
    })


def extract_code(text):
    """Pull the first ```python...``` block from the response."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def build_prompt_with_history(user_message, history):
    """Build a single prompt string that includes prior turns."""
    parts = [SYSTEM_PROMPT, ""]
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        parts.append(f"{role}: {turn['content']}")
    parts.append(f"User: {user_message}")
    return "\n\n".join(parts)


def call_claude_cli(prompt, cli_path):
    """Run claude CLI in print mode and return stdout."""
    result = subprocess.run(
        [cli_path, "-p", "--output-format", "text"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        err = result.stderr.strip() or "claude CLI returned non-zero exit code"
        raise RuntimeError(err)
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class ClaudeProperties(PropertyGroup):
    prompt: StringProperty(
        name="Prompt",
        description="What do you want Claude to do in Blender?",
        default="",
    )
    response: StringProperty(
        name="Response",
        default="",
    )
    generated_code: StringProperty(
        name="Generated Code",
        default="",
    )
    status: StringProperty(
        name="Status",
        default="Ready",
    )
    include_context: BoolProperty(
        name="Include scene context",
        description="Send object names and locations to Claude for accuracy",
        default=True,
    )


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

BUFFER_NAME = "Claude Prompt"

class CLAUDE_OT_open_buffer(Operator):
    """Create/open a text buffer for long prompts in the Text Editor"""
    bl_idname = "claude.open_buffer"
    bl_label = "Open Text Buffer"

    def execute(self, context):
        if BUFFER_NAME not in bpy.data.texts:
            txt = bpy.data.texts.new(BUFFER_NAME)
            txt.write("Type your prompt here, then click 'Send Buffer' in the Claude panel.")
        # Switch any Text Editor area to show the buffer
        for area in context.screen.areas:
            if area.type == 'TEXT_EDITOR':
                area.spaces.active.text = bpy.data.texts[BUFFER_NAME]
                break
        self.report({"INFO"}, f"Edit '{BUFFER_NAME}' in the Text Editor, then click Send Buffer.")
        return {"FINISHED"}


class CLAUDE_OT_send_buffer(Operator):
    """Send the contents of the Claude Prompt text buffer to Claude"""
    bl_idname = "claude.send_buffer"
    bl_label = "Send Buffer"

    def execute(self, context):
        global _is_loading, _pending_result, _conversation_history

        if BUFFER_NAME not in bpy.data.texts:
            self.report({"ERROR"}, "No text buffer found. Click 'Open Text Buffer' first.")
            return {"CANCELLED"}

        prompt = bpy.data.texts[BUFFER_NAME].as_string().strip()
        if not prompt or prompt.startswith("Type your prompt here"):
            self.report({"WARNING"}, "Write your prompt in the text buffer first.")
            return {"CANCELLED"}

        return _do_send(self, context, prompt)


class CLAUDE_OT_send(Operator):
    """Send prompt to Claude and generate Blender Python code"""
    bl_idname = "claude.send"
    bl_label = "Ask Claude"

    def execute(self, context):
        props = context.scene.claude_props
        prompt = props.prompt.strip()

        if not prompt:
            self.report({"WARNING"}, "Enter a prompt first.")
            return {"CANCELLED"}

        return _do_send(self, context, prompt)


def _do_send(operator, context, prompt):
    global _is_loading, _pending_result, _conversation_history

    if _is_loading:
        operator.report({"WARNING"}, "Already waiting for a response.")
        return {"CANCELLED"}

    cli_path = find_claude_cli()
    if not cli_path:
        operator.report({"ERROR"}, "claude CLI not found. Is Claude Code installed?")
        return {"CANCELLED"}

    props = context.scene.claude_props

    # Build user message (optionally with scene context)
    if props.include_context:
        user_content = f"Scene: {get_scene_context()}\n\nTask: {prompt}"
    else:
        user_content = prompt

    _is_loading = True
    props.status = "Thinking..."
    props.response = ""
    props.generated_code = ""

    history_snapshot = list(_conversation_history)
    full_prompt = build_prompt_with_history(user_content, history_snapshot)

    def worker():
        global _is_loading, _pending_result, _conversation_history
        try:
            response_text = call_claude_cli(full_prompt, cli_path)
            code = extract_code(response_text)
            _conversation_history = history_snapshot + [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response_text},
            ]
            _pending_result = (response_text, code, None)
        except Exception as e:
            _pending_result = ("", "", str(e))
        finally:
            _is_loading = False

    threading.Thread(target=worker, daemon=True).start()
    bpy.app.timers.register(_poll_result, first_interval=0.2)
    props.prompt = ""
    return {"FINISHED"}


class CLAUDE_OT_execute(Operator):
    """Execute the generated code inside Blender"""
    bl_idname = "claude.execute"
    bl_label = "Execute Code"

    def execute(self, context):
        code = context.scene.claude_props.generated_code.strip()
        if not code:
            self.report({"WARNING"}, "No code to execute.")
            return {"CANCELLED"}
        try:
            exec(compile(code, "<claude_code>", "exec"), {"bpy": bpy})
            context.scene.claude_props.status = "Executed successfully"
            self.report({"INFO"}, "Code executed")
        except Exception as e:
            context.scene.claude_props.status = f"Error: {e}"
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class CLAUDE_OT_clear(Operator):
    """Clear conversation history and reset the panel"""
    bl_idname = "claude.clear"
    bl_label = "Clear"

    def execute(self, context):
        global _conversation_history
        _conversation_history = []
        props = context.scene.claude_props
        props.response = ""
        props.generated_code = ""
        props.status = "Ready"
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Timer — picks up thread result on the main thread
# ---------------------------------------------------------------------------

def _poll_result():
    global _pending_result
    if _pending_result is None:
        return 0.2  # keep polling

    response_text, code, error = _pending_result
    _pending_result = None

    for scene in bpy.data.scenes:
        if hasattr(scene, "claude_props"):
            if error:
                scene.claude_props.status = f"Error: {error[:120]}"
                scene.claude_props.response = f"Error:\n{error}"
            else:
                scene.claude_props.response = response_text
                scene.claude_props.generated_code = code
                scene.claude_props.status = (
                    "Done — review the code, then click Execute"
                    if code else "Done (no code block found)"
                )
            break

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()

    return None  # unregister timer


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class CLAUDE_PT_main(Panel):
    bl_label = "Claude AI"
    bl_idname = "CLAUDE_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Claude"

    def draw(self, context):
        layout = self.layout
        props = context.scene.claude_props

        # CLI status
        cli_path = find_claude_cli()
        if cli_path:
            layout.label(text="Claude Code: connected", icon="CHECKMARK")
        else:
            layout.label(text="Claude Code CLI not found", icon="ERROR")
            return

        layout.separator()

        # Short prompt input
        layout.label(text="Quick prompt:")
        layout.prop(props, "prompt", text="")

        row = layout.row(align=True)
        row.scale_y = 1.4
        row.enabled = not _is_loading
        row.operator("claude.send", text="Ask Claude", icon="PLAY")
        row.operator("claude.clear", text="", icon="TRASH")

        # Long prompt via text buffer
        layout.separator()
        layout.label(text="Long prompt (paste here):")
        row = layout.row(align=True)
        row.operator("claude.open_buffer", text="Open Text Buffer", icon="TEXT")
        row.operator("claude.send_buffer", text="Send Buffer", icon="PLAY")
        layout.prop(props, "include_context")

        # Status
        layout.separator()
        box = layout.box()
        icon = "SORTTIME" if _is_loading else "INFO"
        box.label(text=props.status, icon=icon)

        # Execute button — right after status, always visible
        if props.generated_code:
            row = layout.row()
            row.scale_y = 1.6
            row.operator("claude.execute", text="Execute Code", icon="CHECKMARK")

        # Code preview (compact)
        if props.generated_code:
            layout.separator()
            layout.label(text="Generated Code:", icon="SCRIPT")
            box = layout.box()
            for line in props.generated_code.split("\n")[:6]:
                box.label(text=line[:80])
            if props.generated_code.count("\n") > 6:
                remaining = props.generated_code.count("\n") - 6
                box.label(text=f"  ... ({remaining} more lines)")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

CLASSES = [
    ClaudeProperties,
    CLAUDE_OT_open_buffer,
    CLAUDE_OT_send_buffer,
    CLAUDE_OT_send,
    CLAUDE_OT_execute,
    CLAUDE_OT_clear,
    CLAUDE_PT_main,
]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.claude_props = bpy.props.PointerProperty(type=ClaudeProperties)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.claude_props


if __name__ == "__main__":
    register()
