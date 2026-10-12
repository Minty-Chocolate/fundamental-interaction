"""Compile configurable industry definitions into a DuckDB group expression."""

from __future__ import annotations

from pathlib import Path

import yaml


def sic_group_expression(config: dict) -> str:
    grouping = config.get("industry_grouping", {})
    name = grouping.get("name")
    levels = config.get("portfolio", {}).get("sic_level")
    if levels is None:
        return "0"
    # Level 1 has an established, irregular SIC division range mapping.
    # Levels 2–4 use literal SIC prefixes unless a named mapping is supplied.
    if name is None and levels == 1:
        name = "division"
    if name is None:
        name = "sic_prefix"
    if name == "sic_prefix":
        if levels not in (1, 2, 3, 4):
            raise ValueError("portfolio.sic_level must be 1, 2, 3, or 4 for sic_prefix")
        return f"CAST(floor(siccd / power(10, {4 - levels})) AS INTEGER)"
    mapping_path = Path(grouping["mapping_file"])
    groups = yaml.safe_load(mapping_path.read_text())[name]["ranges"]
    cases = " ".join(f"WHEN siccd BETWEEN {group['low']} AND {group['high']} THEN '{group['name']}'" for group in groups)
    return f"CASE {cases} ELSE NULL END"
