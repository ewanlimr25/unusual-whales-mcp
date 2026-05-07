# Project Rules: Unusual Whales MCP

This directory contains **project-specific rules** that extend the common rules in `~/.claude/rules/common/` and `~/.claude/rules/python/`.

## Structure

```
.claude/rules/
├── README.md                  # This file
├── mcp-server-patterns.md     # How to write MCP servers
├── data-pipeline.md           # DuckDB/Parquet conventions
└── financial-security.md      # Sensitive financial data handling
```

## How These Rules Apply

**Priority**: Project rules > Python rules > Common rules

When a project rule conflicts with a global rule, the project rule takes precedence. For example:
- Common rule: "Limit functions to 50 lines"
- Project rule: "MCP tool functions can be up to 100 lines if they handle multiple related sub-analyses"

The project rule applies here.

## Quick Links

- **MCP Server Development**: See `mcp-server-patterns.md`
- **Data Access**: See `data-pipeline.md`
- **Security**: See `financial-security.md`
- **Full Project Docs**: See `../CLAUDE.md`

## Why These Rules Exist

1. **mcp-server-patterns.md** — MCP servers have a specific structure (list_tools, call_tool, stdio_server). Codifying the pattern ensures consistency across all 10 servers.

2. **data-pipeline.md** — DuckDB + Parquet is the core data stack. Conventions around column selection, row limits, and result formatting ensure performance and consistency.

3. **financial-security.md** — Trading data is sensitive. This rule ensures we never accidentally expose raw trades, leak timestamps, or hardcode data paths.

---

**Last updated**: 2026-05-07
