"""Frozen context budget measured for the exact THI3-41 runtime route."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RuntimeEnvelopeBudget:
    context_window_tokens: int
    fixed_system_and_bridge_tokens: int
    working_memory_tokens: int
    output_and_reasoning_tokens: int
    absolute_transcript_tokens: int
    unallocated_safety_tokens: int
    routine_batch_target_tokens: int

    @classmethod
    def pinned(cls) -> "RuntimeEnvelopeBudget":
        return cls(
            context_window_tokens=272_000,
            fixed_system_and_bridge_tokens=791,
            working_memory_tokens=16_000,
            output_and_reasoning_tokens=32_768,
            absolute_transcript_tokens=200_000,
            unallocated_safety_tokens=22_441,
            routine_batch_target_tokens=8_000,
        )

    @property
    def measured_residual_tokens(self) -> int:
        return self.context_window_tokens - (
            self.fixed_system_and_bridge_tokens
            + self.working_memory_tokens
            + self.output_and_reasoning_tokens
        )

    @property
    def total_reserved_tokens(self) -> int:
        return (
            self.fixed_system_and_bridge_tokens
            + self.working_memory_tokens
            + self.output_and_reasoning_tokens
            + self.absolute_transcript_tokens
            + self.unallocated_safety_tokens
        )

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


__all__ = ["RuntimeEnvelopeBudget"]
