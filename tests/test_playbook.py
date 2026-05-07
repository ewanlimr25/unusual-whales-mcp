"""Tests for servers.playbook (renamed from strategy).

Holds suggest_strategy + batch_strategy_scan (formerly in uw-strategy) and
daily_synthesis (moved from uw-insights). Server is uw-playbook.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_server_lists_three_tools():
    """uw-playbook exposes suggest_strategy + batch_strategy_scan + daily_synthesis."""
    from servers.playbook import list_tools

    tools = await list_tools()
    names = {t.name for t in tools}
    assert names == {"suggest_strategy", "batch_strategy_scan", "daily_synthesis"}


@pytest.mark.asyncio
async def test_server_name_is_uw_playbook():
    from servers.playbook import server

    assert server.name == "uw-playbook"


@pytest.mark.asyncio
async def test_daily_synthesis_dispatches_with_no_data(stocks_dir: Path):
    """No Parquet files in the tmp stocks_dir → each subcomponent returns an
    error dict; daily_synthesis must wrap them, not crash."""
    from servers.playbook import call_tool

    result = await call_tool("daily_synthesis", {})
    payload = json.loads(result[0].text)
    # Expected structure regardless of upstream errors
    for key in ("date", "market_regime", "top_bullish_signals",
                "top_bearish_signals", "watchlist_alerts"):
        assert key in payload


@pytest.mark.asyncio
async def test_suggest_strategy_dispatches(stocks_dir: Path):
    """Smoke test that the moved tool still dispatches under the new server name."""
    from servers.playbook import call_tool

    # No data → graceful empty signals
    result = await call_tool("suggest_strategy", {"symbol": "FAKE"})
    payload = json.loads(result[0].text)
    assert payload["symbol"] == "FAKE"


@pytest.mark.asyncio
async def test_old_strategy_module_no_longer_referenced():
    """The renamed file should be the source of truth — no lingering import
    of the old `servers.strategy` module from production code."""
    import importlib.util
    spec = importlib.util.find_spec("servers.strategy")
    assert spec is None, "servers.strategy should be removed in favor of servers.playbook"
