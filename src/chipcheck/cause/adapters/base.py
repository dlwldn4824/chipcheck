from __future__ import annotations

from typing import Callable, Protocol

from ..schema import DefectEvent

REGISTRY: dict[str, "LogAdapter"] = {}


class LogAdapter(Protocol):
    """구현해야 할 것은 이 메서드 하나다."""

    name: str

    def parse(self, raw: str) -> list[DefectEvent]: ...


def register(adapter: "LogAdapter") -> "LogAdapter":
    REGISTRY[adapter.name] = adapter
    return adapter


def get(name: str) -> "LogAdapter":
    if name not in REGISTRY:
        raise KeyError(f"어댑터 없음: {name}. 등록된 것: {sorted(REGISTRY)}")
    return REGISTRY[name]
