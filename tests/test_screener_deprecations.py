"""Tests for v0.4.0 deprecations in uw-screener (Phase 4)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_put_call_ratio_extremes_marked_deprecated():
    """The screener tool description must lead with [DEPRECATED v0.4.0]
    pointing to pc_ratio_zscore. Audit §1.3: raw P/C is strictly dominated
    by the statistical z-score.
    """
    from servers.screener import list_tools

    tools = await list_tools()
    pcr = next(t for t in tools if t.name == "put_call_ratio_extremes")
    assert pcr.description.startswith("[DEPRECATED v0.4.0]")
    assert "pc_ratio_zscore" in pcr.description


@pytest.mark.asyncio
async def test_put_call_ratio_extremes_still_functional():
    """Deprecated, not removed — must still respond cleanly."""
    from servers.screener import call_tool

    # No data fixture → graceful handling (FileNotFoundError or empty result).
    # We just verify it dispatches without raising.
    try:
        await call_tool("put_call_ratio_extremes", {"top_n": 1})
    except FileNotFoundError:
        # Acceptable when no Parquet data is present
        pass
