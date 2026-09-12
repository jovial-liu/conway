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

    def decide(self, constitution: str, memory: str, observation: Observation) -> Decision:
        instruction = f"""You are Conway's decision core. There is no chat operator waiting to give you a next instruction.

CURRENT CONTEXT:\n{memory}\n
OBSERVATION METADATA:
- screen: {observation.width}x{observation.height}
- cursor: ({observation.cursor_x}, {observation.cursor_y})

Inspect the screenshot and choose exactly one next GUI action that best advances the objectives in the constitution. Verify prior results from the context before retrying an action.

Return JSON only with this exact shape:
{{
  "action": {{"type": "wait|click|double_click|move|type|press|hotkey|scroll|drag", "args": {{}}}},
  "rationale": "one concise sentence",
  "memory_note": "optional concise durable fact, or null"
}}

Coordinates are absolute screenshot pixels. click/double_click/move/drag use args.x and args.y. hotkey uses args.keys. press uses args.key. type uses args.text. scroll uses args.amount. wait may use args.seconds. If the state is ambiguous or no useful action is justified, choose wait. Do not output private chain-of-thought."""

        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 512,
            "messages": [
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
        }
        response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        raw = self._content_text(data["choices"][0]["message"]["content"])
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


class MockBrain(Brain):
    def decide(self, constitution: str, memory: str, observation: Observation) -> Decision:
        return Decision(
            action={"type": "wait", "args": {"seconds": 0.1}},
            rationale="Mock brain is active; no model action was requested.",
        )
