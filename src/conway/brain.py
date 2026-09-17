from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import httpx

from .computer import Observation
from .opencua import parse_opencua_action


@dataclass(slots=True)
class Decision:
    action: dict[str, Any]
    rationale: str = ""
    memory_note: str | None = None


class Brain:
    def decide(self, constitution: str, memory: str, observation: Observation) -> Decision:
        raise NotImplementedError

    def compact_memory(self, constitution: str, memory: str) -> str | None:
        """Optionally rewrite accumulated file memory into a compact rolling summary."""
        return None

    def close(self) -> None:
        return None


class OpenAICompatibleVLM(Brain):
    """Multimodal brain for OpenAI-compatible local servers such as llama.cpp/vLLM."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8042/v1",
        model: str = "local-model",
        timeout: float = 180.0,
        tool_manifest: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tool_manifest = tool_manifest or '- wait: {"seconds": 1}'
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _image_data_url(path: Path) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            return "\n".join(chunks)
        return str(content)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("model JSON response must be an object")
            return parsed
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(text[start : end + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("model JSON response must be an object")
                return parsed
            raise

    def _chat(self, messages: list[dict[str, Any]], *, max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                data = response.json()
                return self._content_text(data["choices"][0]["message"]["content"])
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt >= 2:
                    break
                time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"Model request failed after retries: {last_error}") from last_error

    def decide(self, constitution: str, memory: str, observation: Observation) -> Decision:
        active = observation.active_app or "unknown"
        window = observation.active_window or "unknown"
        instruction = f"""You are Conway's decision core. There is no chat operator waiting to give you a next instruction.

CURRENT CONTEXT:\n{memory}\n
OBSERVATION METADATA:
- screenshot pixels shown to you: {observation.width}x{observation.height}
- OS input coordinate space: {observation.input_width}x{observation.input_height}
- screenshot cursor: ({observation.cursor_x}, {observation.cursor_y})
- active app: {active}
- active window: {window}

Choose exactly one next action that best advances the constitution. Prefer a direct system/file tool when it is more reliable than manipulating the same thing through a GUI. Use screenshot-driven GUI actions when the task is only available through the interface or visual state matters. Verify prior results from context before retrying an action.

AVAILABLE ACTIONS:
{self.tool_manifest}

Return JSON only with this exact shape:
{{
  "action": {{"type": "one enabled action", "args": {{}}}},
  "rationale": "one concise sentence",
  "memory_note": "optional concise durable fact, or null"
}}

For GUI coordinates, use absolute pixels in the screenshot you received, not OS logical coordinates. Conway maps screenshot pixels to the host input coordinate system. If the state is ambiguous or no useful action is justified, choose wait. Do not output private chain-of-thought."""

        raw = self._chat(
            [
                {"role": "system", "content": constitution},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_data_url(observation.screenshot_path)},
                        },
                    ],
                },
            ],
            max_tokens=700,
            temperature=0.1,
        )
        parsed = self._extract_json(raw)
        action = parsed.get("action") or {"type": "wait", "args": {"seconds": 1}}
        if not isinstance(action, dict):
            raise ValueError("model returned a non-object action")
        memory_note = parsed.get("memory_note")
        if memory_note is not None:
            memory_note = str(memory_note).strip() or None
        return Decision(
            action=action,
            rationale=str(parsed.get("rationale", "")).strip(),
            memory_note=memory_note,
        )

    def compact_memory(self, constitution: str, memory: str) -> str | None:
        prompt = f"""Rewrite Conway's accumulated memory into a compact rolling Markdown state.

Preserve only information likely to matter in future cycles:
- current or long-term objectives that are actually in progress;
- durable environment facts and useful discoveries;
- unfinished work and important dependencies;
- recurring failure modes or strategies that should not be repeated;
- concise state needed to continue a multi-step task.

Drop transient cursor coordinates, old timestamps, duplicated facts, verbose rationales, and completed low-value steps. Do not invent facts. Keep it under roughly 1200 words.

MEMORY TO COMPACT:\n{memory}
"""
        text = self._chat(
            [
                {"role": "system", "content": constitution},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1600,
            temperature=0.0,
        ).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text or None


class OpenCUABrain(OpenAICompatibleVLM):
    """Adapter for OpenCUA's native pyautogui-style grounding output."""

    def decide(self, constitution: str, memory: str, observation: Observation) -> Decision:
        active = observation.active_app or "unknown"
        window = observation.active_window or "unknown"
        instruction = f"""Act as an autonomous computer-use agent under the constitution.

CURRENT CONTEXT:\n{memory}\n
CURRENT DESKTOP:
- active app: {active}
- active window: {window}

Inspect the screenshot and output exactly one pyautogui action in your native OpenCUA coordinate convention, for example:
pyautogui.click(x=960, y=324)

Use only one of click, doubleClick, moveTo, dragTo, write, press, hotkey, scroll, or sleep. Do not emit prose or executable multi-line Python."""
        raw = self._chat(
            [
                {"role": "system", "content": constitution},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_data_url(observation.screenshot_path)},
                        },
                    ],
                },
            ],
            max_tokens=256,
            temperature=0.0,
        )
        action = parse_opencua_action(raw, observation.width, observation.height)
        return Decision(action=action, rationale=f"OpenCUA selected {action['type']}.")

    def compact_memory(self, constitution: str, memory: str) -> str | None:
        # OpenCUA is optimized for GUI grounding rather than durable-state summarization.
        return None


class MockBrain(Brain):
    def decide(self, constitution: str, memory: str, observation: Observation) -> Decision:
        return Decision(
            action={"type": "wait", "args": {"seconds": 0.1}},
            rationale="Mock brain is active; no model action was requested.",
        )
