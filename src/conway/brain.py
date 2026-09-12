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
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=timeout)

    @staticmethod
    def _image_data_url(path: Path) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

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
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def decide(self, constitution: str, memory: str, observation: Observation) -> Decision:
        instruction = f"""You are the decision core of Conway.

CONSTITUTION:\n{constitution}\n
CURRENT MEMORY:\n{memory}\n
CURRENT SCREEN SIZE: {observation.width}x{observation.height}

Look at the screenshot and choose exactly one next GUI action. Do not wait for a human prompt.
Return JSON only, using this schema:
{{
  "action": {{"type": "wait|click|double_click|move|type|press|hotkey|scroll|drag", "args": {{}}}},
  "rationale": "one concise sentence",
  "memory_note": "optional concise durable fact, or null"
}}

For click/double_click/move/drag use absolute screenshot coordinates x/y. For hotkey use args.keys as a list. For press use args.key. For typing use args.text. For scroll use args.amount. If no useful action is justified, use wait. Do not output hidden chain-of-thought."""

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_data_url(observation.screenshot_path)},
                        },
                    ],
                }
            ],
        }
        response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        raw = data["choices"][0]["message"]["content"]
        parsed = self._extract_json(raw)
        action = parsed.get("action") or {"type": "wait", "args": {"seconds": 1}}
        if not isinstance(action, dict):
            raise ValueError("Model returned a non-object action")
        return Decision(
            action=action,
            rationale=str(parsed.get("rationale", "")),
            memory_note=parsed.get("memory_note"),
        )


class MockBrain(Brain):
    def decide(self, constitution: str, memory: str, observation: Observation) -> Decision:
        return Decision(
            action={"type": "wait", "args": {"seconds": 1}},
            rationale="Mock brain is active; no model action was executed.",
        )
