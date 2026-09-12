from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import httpx

from .computer import Observation


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


class OpenAICompatibleVLM(Brain):
    """Multimodal brain for local OpenAI-compatible servers such as llama.cpp/vLLM."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8042/v1",
        model: str = "local-model",
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=timeout)

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
        response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return self._content_text(data["choices"][0]["message"]["content"])

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

Inspect the screenshot and choose exactly one next GUI action that best advances the objectives in the constitution. Verify prior results from the context before retrying an action.

Return JSON only with this exact shape:
{{
  "action": {{"type": "wait|click|double_click|move|type|press|hotkey|scroll|drag", "args": {{}}}},
  "rationale": "one concise sentence",
  "memory_note": "optional concise durable fact, or null"
}}

Coordinates MUST be absolute pixels in the screenshot you received, not OS logical coordinates. Conway will map screenshot pixels to the host input coordinate system. click/double_click/move/drag use args.x and args.y. hotkey uses args.keys. press uses args.key. type uses args.text. scroll uses args.amount. wait may use args.seconds. If the state is ambiguous or no useful action is justified, choose wait. Do not output private chain-of-thought."""

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
            max_tokens=512,
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


class MockBrain(Brain):
    def decide(self, constitution: str, memory: str, observation: Observation) -> Decision:
        return Decision(
            action={"type": "wait", "args": {"seconds": 0.1}},
            rationale="Mock brain is active; no model action was requested.",
        )
