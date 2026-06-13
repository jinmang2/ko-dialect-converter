from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SYSTEM_PROMPT = "당신은 한국어 방언 변환 전문가입니다."

DO_NAME: dict[str, str] = {
    "gangwondo": "강원도",
    "gyeongsangdo": "경상도",
}

Direction = Literal["std2dia", "dia2std"]


@dataclass
class ChatTemplate:
    system: str = SYSTEM_PROMPT

    def format_user(self, source: str, do: str, direction: Direction) -> str:
        do_name = DO_NAME.get(do, do)
        if direction == "std2dia":
            return f"다음 문장을 {do_name} 사투리로 바꿔줘:\n{source}"
        return f"다음 {do_name} 사투리를 표준어로 바꿔줘:\n{source}"

    def format_messages(
        self,
        source: str,
        target: str,
        do: str,
        direction: Direction,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.format_user(source, do, direction)},
            {"role": "assistant", "content": target},
        ]

    def apply(
        self,
        tokenizer,
        source: str,
        target: str,
        do: str,
        direction: Direction,
        add_generation_prompt: bool = False,
    ) -> str:
        messages = self.format_messages(source, target, do, direction)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def build_prompt(self, tokenizer, source: str, do: str, direction: Direction) -> str:
        """Prompt only (no assistant turn) — for GRPO/evaluation."""
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.format_user(source, do, direction)},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# --------------------------------------------------------------------------- #
# Template registry
# --------------------------------------------------------------------------- #
# A prosody-aware system prompt variant (see IDEAS.md): nudges the model to keep
# sentence-final endings/intonation markers when converting toward dialect.
PROSODY_SYSTEM_PROMPT = (
    "당신은 한국어 방언 변환 전문가입니다. "
    "문장의 어미와 억양, 종결 표현을 자연스럽게 살려서 변환하세요."
)

TEMPLATE_REGISTRY: dict[str, ChatTemplate] = {
    "default": ChatTemplate(),
    "prosody": ChatTemplate(system=PROSODY_SYSTEM_PROMPT),
}


def register_template(name: str, template: ChatTemplate) -> None:
    """Register (or override) a named :class:`ChatTemplate` for config-driven selection."""
    TEMPLATE_REGISTRY[name] = template


def get_template(name: str = "default") -> ChatTemplate:
    """Resolve a registered template by name.

    Swapping the chat template is a config knob (``template_name``); the dataset no
    longer has to be rebuilt to try a new one (see ``output_mode='structured'``).
    """
    try:
        return TEMPLATE_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown template {name!r}. Registered: {sorted(TEMPLATE_REGISTRY)}."
        ) from None
