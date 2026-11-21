"""Locate the bundled physical model and user-supplied single-domain size limits."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def base_root():
    return ROOT / "portable_runtime" / "STAGE6_E1_WP0_WP1_2026-10-22"


def size_table():
    """Read the two critical-size curves from a local SDCC workbook or its JSON export."""
    filename = os.environ.get("MAGTHERMO_SD_TABLE")
    if not filename:
        raise FileNotFoundError("Set MAGTHERMO_SD_TABLE to TM_Crit_Sizes.xls or its JSON export.")
    path = Path(filename)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() != ".xls":
        raise ValueError("Expected an Excel .xls workbook or its JSON export.")
    import xlrd

    with xlrd.open_workbook(str(path), on_demand=True) as workbook:
        tables = {}
        for name in ("x_Hard DOWN", "x_Easy DOWN"):
            sheet = workbook.sheet_by_name(name)
            tables[name] = [sheet.row_values(i) for i in range(sheet.nrows)]
    return tables
