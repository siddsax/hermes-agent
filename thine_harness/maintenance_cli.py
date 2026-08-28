"""Mac-local operator CLI for Hermes authoritative state and maintenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from .home_state import HomeStateProjector, default_database_path as home_database_path
from .maintenance import AuthoritativeStateReader, ResetScope, RetentionResetService
from .run_state import DurableRunState, default_database_path as run_database_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    user_id = str(args.user_id)
    state = DurableRunState(Path(args.state_db))
    home = HomeStateProjector(Path(args.home_db))
    service = RetentionResetService(state, home=home)

    if args.command == "inspect":
        _print(AuthoritativeStateReader(state, home=home).snapshot(user_id))
        return 0
    if args.command == "debug":
        _print(
            AuthoritativeStateReader(state, home=home).debug_invocations(
                user_id, limit=int(args.limit)
            )
        )
        return 0
    if args.command == "retention-policy":
        _print(service.retention_policy())
        return 0
    if args.command == "retention-cleanup":
        expected = f"CLEANUP {user_id}"
        if args.confirm != expected:
            parser.error(f"retention cleanup requires --confirm {expected!r}")
        _print(service.cleanup(user_id).to_dict())
        return 0
    if args.command == "reset-plan":
        _print(service.plan_reset(user_id, cast(ResetScope, args.scope)).to_dict())
        return 0
    if args.command == "reset-execute":
        result = service.execute_reset(
            reset_id=str(args.reset_id),
            confirmation=str(args.confirm),
            harness_stopped=bool(args.harness_stopped),
        )
        _print(result.to_dict())
        return 0 if result.payload.status == "completed" else 2
    raise AssertionError(f"unknown command {args.command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m thine_harness.maintenance_cli",
        description=(
            "Inspect authoritative local Hermes state or run confirmation-bound "
            "retention/reset maintenance. This never opens Thine backend storage."
        ),
    )
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--state-db", default=str(run_database_path()))
    parser.add_argument("--home-db", default=str(home_database_path()))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect")
    debug = commands.add_parser("debug")
    debug.add_argument("--limit", type=int, choices=range(1, 51), default=50)
    commands.add_parser("retention-policy")
    cleanup = commands.add_parser("retention-cleanup")
    cleanup.add_argument("--confirm", required=True)
    plan = commands.add_parser("reset-plan")
    plan.add_argument(
        "--scope",
        required=True,
        choices=(
            "working_memory_topics",
            "queues_schedules_receipts",
            "home_state",
            "all_hermes_state",
        ),
    )
    execute = commands.add_parser("reset-execute")
    execute.add_argument("--reset-id", required=True)
    execute.add_argument("--confirm", required=True)
    execute.add_argument(
        "--harness-stopped",
        action="store_true",
        help="Operator assertion; live lease/claim/finalization checks still run.",
    )
    return parser


def _print(value: object) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())


__all__ = ["main"]
