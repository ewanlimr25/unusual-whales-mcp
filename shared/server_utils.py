"""Common utilities for building MCP servers."""

import json

from mcp.types import TextContent


def text_result(data) -> list[TextContent]:
    """Convert data to a TextContent response."""
    if isinstance(data, str):
        text = data
    else:
        text = json.dumps(data, indent=2, default=str)
    return [TextContent(type="text", text=text)]


def df_to_result(df, orient="records", max_rows=50) -> list[TextContent]:
    """Convert a DataFrame to a TextContent response, capped at max_rows."""
    if len(df) > max_rows:
        note = f"(showing top {max_rows} of {len(df)} rows)\n"
    else:
        note = ""
    text = note + df.head(max_rows).to_json(orient=orient, indent=2, default_handler=str)
    return [TextContent(type="text", text=text)]
