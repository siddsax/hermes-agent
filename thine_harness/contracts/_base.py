"""Typed, immutable wrappers for the language-neutral v1 wire payloads."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, FrozenInstanceError
import json
from types import MappingProxyType
from typing import Any, ClassVar, Generic, TypeVar, cast


JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]

_DTO_BY_TARGET: dict[str, type["ContractDTO"]] = {}


@dataclass(init=False, slots=True, eq=False)
class FrozenJSONObject(Mapping[str, Any]):
    """Read-only object view with both mapping and typed attribute access."""

    _values: Mapping[str, FrozenJSONValue]

    def __init__(self, values: dict[str, FrozenJSONValue]):
        object.__setattr__(self, "_values", MappingProxyType(values))

    def __getitem__(self, key: str) -> FrozenJSONValue:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> FrozenJSONValue:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_values" and not hasattr(self, "_values"):
            object.__setattr__(self, name, value)
            return
        raise FrozenInstanceError(f"cannot assign to field {name!r}")


FrozenJSONValue = (
    None | bool | int | float | str | tuple["FrozenJSONValue", ...] | FrozenJSONObject
)


Payload = TypeVar("Payload")


@dataclass(init=False, slots=True)
class ContractDTO(Generic[Payload]):
    """One validated wire payload with a contract-specific Python type."""

    type_id: ClassVar[str]
    _optional_fields: ClassVar[Mapping[tuple[str, ...], frozenset[str]]] = {}
    _wire_json: str

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

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
            raise TypeError(
                f"decoder returned {type(decoded).__name__}, expected {cls.__name__}"
            )
        return cast(DTO, decoded)

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

    @property
    def payload(self) -> Payload:
        value = json.loads(self._wire_json)
        return cast(Payload, _freeze_json(value, self._optional_fields))


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


def _freeze_json(
    value: JSONValue,
    optional_fields: Mapping[tuple[str, ...], frozenset[str]],
    path: tuple[str, ...] = (),
) -> FrozenJSONValue:
    if isinstance(value, dict):
        frozen = {
            key: _freeze_json(child, optional_fields, (*path, key))
            for key, child in value.items()
        }
        for optional_field in optional_fields.get(path, ()):
            frozen.setdefault(optional_field, None)
        return FrozenJSONObject(frozen)
    if isinstance(value, list):
        return tuple(
            _freeze_json(child, optional_fields, (*path, "*")) for child in value
        )
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value
