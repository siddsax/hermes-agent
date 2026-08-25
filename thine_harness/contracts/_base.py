"""Typed, immutable wrappers for the language-neutral v1 wire payloads."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import ClassVar, TypeVar


JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]

_DTO_BY_TARGET: dict[str, type["ContractDTO"]] = {}


@dataclass(frozen=True, init=False)
class ContractDTO:
    """One validated wire payload with a contract-specific Python type."""

    type_id: ClassVar[str]
    _wire_json: str

    @classmethod
    def from_dict(cls: type[DTO], payload: dict[str, JSONValue]) -> DTO:
        from .codec import validate_payload

        validated = validate_payload(cls.type_id, payload)
        return cls._from_validated(validated)

    @classmethod
    def from_json(cls: type[DTO], wire: str | bytes | bytearray) -> DTO:
        from .codec import decode_contract

        decoded = decode_contract(cls.type_id, wire)
        if not isinstance(decoded, cls):
            raise TypeError(f"decoder returned {type(decoded).__name__}, expected {cls.__name__}")
        return decoded

    @classmethod
    def _from_validated(cls: type[DTO], payload: dict[str, JSONValue]) -> DTO:
        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "_wire_json",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
        return instance

    def to_dict(self) -> dict[str, JSONValue]:
        value = json.loads(self._wire_json)
        if not isinstance(value, dict):
            raise TypeError("validated contract payload is not an object")
        return value

    def to_json(self) -> str:
        return self._wire_json


DTO = TypeVar("DTO", bound=ContractDTO)


def contract_type(type_id: str):
    """Register one explicit DTO class for a manifest target."""

    def decorate(cls: type[DTO]) -> type[DTO]:
        if type_id in _DTO_BY_TARGET:
            raise RuntimeError(f"duplicate Python DTO registration for {type_id!r}")
        cls.type_id = type_id
        _DTO_BY_TARGET[type_id] = cls
        return cls

    return decorate
