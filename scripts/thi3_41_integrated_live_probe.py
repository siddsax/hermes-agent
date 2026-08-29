"""Cross-platform launcher for the checked-in THI3-41 integrated proof."""

from __future__ import annotations

from pathlib import Path
import sys


WORKTREE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

from thine_harness.integrated_probe import main  # noqa: E402


raise SystemExit(main())
