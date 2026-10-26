"""Minimal LLM client returning schema-valid JSON.

Providers (chosen per model profile in the [tool.patsf] section of pyproject.toml):
  openai_compatible  Ollama, llama.cpp server, vLLM or a hosted open-weight service
  anthropic          Claude, as an optional reference model (needs ANTHROPIC_API_KEY)
  rules              no model: the rule-based baseline in rules.py (spaCy)
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path
from typing import Protocol

from .schema import schema_errors as _validate


class LLM(Protocol):
    name: str

    def json(self, system: str, user: str, schema: dict, tool_name: str) -> dict: ...


def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # reasoning models such as Qwen3
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    return json.loads(text[start:end + 1])


class OpenAICompatibleLLM:
    """Ollama, llama.cpp server, vLLM or any OpenAI-compatible endpoint.

    Asks the server to enforce the JSON Schema (response_format json_schema). Servers
    that do not support it fall back to plain JSON mode.
    """

    def __init__(self, model: str, base_url: str, temperature: float = 0.0,
                 reasoning: str | None = None, timeout: float = 3600):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "none"),
                             timeout=timeout)
        self.name = f"openai_compatible:{model}"
        self.model, self.temperature = model, temperature
        self.reasoning = reasoning if "gpt-oss" in model else None
        self.schema_mode = True

    def _create(self, messages, schema, tool_name):
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature)
        if self.schema_mode:
            try:
                return self.client.chat.completions.create(
                    **kwargs, response_format={"type": "json_schema",
                                               "json_schema": {"name": tool_name, "schema": schema, "strict": True}})
            except Exception as exc:  # server rejects json_schema mode: fall back to json_object
                if getattr(exc, "status_code", None) not in (400, 404, 422):
                    raise
                self.schema_mode = False
        return self.client.chat.completions.create(**kwargs, response_format={"type": "json_object"})

    def json(self, system, user, schema, tool_name):
        system_text = (f"Reasoning: {self.reasoning}\n\n" if self.reasoning else "") + system
        system_text += "\n\nReturn only a JSON object that validates against this JSON Schema:\n" + json.dumps(schema)
        messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user}]
        errors: list[str] = []
        for _ in range(3):
            resp = self._create(messages, schema, tool_name)
            text = resp.choices[0].message.content or ""
            try:
                obj = _parse_json(text)
                errors = _validate(obj, schema)
            except json.JSONDecodeError as exc:
                obj, errors = None, [f"invalid JSON: {exc}"]
            if not errors:
                return obj
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "Schema errors, resubmit the full JSON:\n" + "\n".join(errors[:20])}]
        raise ValueError(f"{tool_name}: output still invalid after retries: {errors[:5]}")


class AnthropicLLM:
    """Claude, for use as a reference model in agreement checks.

    Current Claude models take no sampling parameters (temperature) and no forced tool
    choice, so the result tool is offered with tool_choice "auto", the prompt says to call
    it, and the answer is validated against the schema here, with retries.
    """

    def __init__(self, model: str, max_tokens: int = 64000, effort: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.name = f"anthropic:{model}"
        self.model, self.max_tokens, self.effort = model, max_tokens, effort

    def json(self, system, user, schema, tool_name):
        system = system + f"\n\nSubmit your result by calling the {tool_name} tool exactly once."
        tools = [{"name": tool_name, "description": "Submit the result.", "input_schema": schema}]
        extra = {"output_config": {"effort": self.effort}} if self.effort else {}
        messages = [{"role": "user", "content": user}]
        errors: list[str] = []
        for _ in range(3):
            with self.client.messages.stream(model=self.model, max_tokens=self.max_tokens, system=system,
                                             messages=messages, tools=tools,
                                             tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                                             **extra) as stream:
                resp = stream.get_final_message()
            calls = [b for b in resp.content if b.type == "tool_use"]
            messages.append({"role": "assistant", "content": resp.content})
            if not calls:
                errors = [f"no {tool_name} call (stop_reason: {resp.stop_reason})"]
                messages.append({"role": "user", "content": f"Call the {tool_name} tool with your result."})
                continue
            results = []
            for b in calls:  # the API needs a tool_result for every tool_use
                if b.name == tool_name and not (errors := _validate(b.input, schema)):
                    return b.input
                results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                "content": "Schema errors, resubmit:\n" + "\n".join(errors[:20])
                                if b.name == tool_name else f"Unknown tool; call {tool_name}."})
            messages.append({"role": "user", "content": results})
        raise ValueError(f"{tool_name}: output still invalid after retries: {errors[:5]}")


def load_profile(name: str | None = None, path: str | Path = "pyproject.toml") -> tuple[str, dict]:
    """The named model profile from pyproject.toml (or its `default`), as (name, profile)."""
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))["tool"]["patsf"]
    name = name or cfg["default"]
    if name not in cfg["models"]:
        raise SystemExit(f"unknown profile {name!r}; profiles in pyproject.toml: {', '.join(cfg['models'])}")
    return name, cfg["models"][name]


def from_profile(profile: dict) -> LLM:
    """Build the client for one [tool.patsf.models.<name>] profile."""
    if profile["provider"] == "rules":
        from .rules import RuleExtractor
        return RuleExtractor(profile.get("spacy_model", "en_core_web_trf"))
    if profile["provider"] == "openai_compatible":
        return OpenAICompatibleLLM(profile["model"], profile["base_url"],
                                   temperature=profile.get("temperature", 0.0),
                                   reasoning=profile.get("reasoning", "medium"))
    return AnthropicLLM(profile["model"], effort=profile.get("effort"))
