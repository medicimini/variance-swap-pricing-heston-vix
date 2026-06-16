"""Data loading and validation utilities for the option surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataValidationReport:
    n_rows: int
    n_maturities: int
    n_strikes: int
    min_maturity: float
    max_maturity: float
    min_strike: float
    max_strike: float
    min_price: float
    max_price: float
    has_target_maturity: bool

    def as_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


def _normalise_column_name(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _column_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    col = 0
    for ch in letters:
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return col - 1


def _read_simple_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    """Read a simple xlsx worksheet with stdlib only.

    This avoids requiring openpyxl in the submitted assignment code. It is deliberately
    small, because the provided workbook contains a flat table on the first sheet.
    """

    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    with zipfile.ZipFile(path) as zf:
        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared_strings.append("".join(t.text or "" for t in item.findall(".//a:t", ns)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        first_sheet = workbook.find("a:sheets/a:sheet", ns)
        rel_id = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]

        rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_target = None
        for rel in rels_root:
            if rel.attrib.get("Id") == rel_id:
                rel_target = rel.attrib["Target"]
                break
        if rel_target is None:
            raise ValueError("Could not locate first worksheet relationship in xlsx file.")
        sheet_path = "xl/" + rel_target.lstrip("/")
        if sheet_path not in zf.namelist():
            sheet_path = "xl/worksheets/sheet1.xml"

        sheet = ET.fromstring(zf.read(sheet_path))
        rows = []
        for row in sheet.findall(".//a:sheetData/a:row", ns):
            values = {}
            max_col = -1
            for cell in row.findall("a:c", ns):
                idx = _column_index(cell.attrib.get("r", "A1"))
                max_col = max(max_col, idx)
                raw_value = cell.find("a:v", ns)
                value = raw_value.text if raw_value is not None else None
                if cell.attrib.get("t") == "s" and value is not None:
                    value = shared_strings[int(value)]
                values[idx] = value
            if max_col >= 0:
                rows.append([values.get(i) for i in range(max_col + 1)])

    if not rows:
        return pd.DataFrame()
    header = rows[0]
    body = rows[1:]
    return pd.DataFrame(body, columns=header)


def load_option_data(path: str | Path) -> pd.DataFrame:
    """Load raw Excel option data and return canonical columns.

    Returned columns are: ``Strike``, ``Maturity`` and ``Price``.
    The assignment data are call prices; no put prices are assumed to be present.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Option data file not found: {path}")

    if path.suffix.lower() == ".xlsx":
        raw = _read_simple_xlsx_first_sheet(path)
    else:
        raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError("The option data workbook is empty.")

    normalised = {_normalise_column_name(c): c for c in raw.columns}
    aliases = {
        "Strike": ["strike", "strikes", "k"],
        "Maturity": ["maturity", "maturities", "t", "time_to_maturity"],
        "Price": ["price", "prices", "call", "call_price", "market_price"],
    }

    selected = {}
    for canonical, names in aliases.items():
        for name in names:
            if name in normalised:
                selected[canonical] = normalised[name]
                break
        if canonical not in selected:
            raise ValueError(
                f"Could not find required column {canonical!r}; available columns are {list(raw.columns)!r}"
            )

    df = raw[[selected["Strike"], selected["Maturity"], selected["Price"]]].copy()
    df.columns = ["Strike", "Maturity", "Price"]
    for col in ["Strike", "Maturity", "Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Strike", "Maturity", "Price"])
    df = df.sort_values(["Maturity", "Strike"]).reset_index(drop=True)
    validate_option_data(df)
    return df


def validate_option_data(df: pd.DataFrame, target_maturity: float = 4.0 / 12.0, tol: float = 1e-8) -> DataValidationReport:
    required = {"Strike", "Maturity", "Price"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Option data missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Option data contain no usable rows.")
    if (df["Strike"] <= 0).any():
        raise ValueError("All strikes must be strictly positive.")
    if (df["Maturity"] <= 0).any():
        raise ValueError("All maturities must be strictly positive.")
    if (df["Price"] < 0).any():
        raise ValueError("Option prices must be non-negative.")

    return DataValidationReport(
        n_rows=int(len(df)),
        n_maturities=int(df["Maturity"].nunique()),
        n_strikes=int(df["Strike"].nunique()),
        min_maturity=float(df["Maturity"].min()),
        max_maturity=float(df["Maturity"].max()),
        min_strike=float(df["Strike"].min()),
        max_strike=float(df["Strike"].max()),
        min_price=float(df["Price"].min()),
        max_price=float(df["Price"].max()),
        has_target_maturity=bool(np.isclose(df["Maturity"], target_maturity, atol=tol).any()),
    )


def maturity_slice(df: pd.DataFrame, maturity: float, tol: float = 1e-8) -> pd.DataFrame:
    """Return rows matching a target maturity."""

    out = df[np.isclose(df["Maturity"], maturity, atol=tol)].copy()
    if out.empty:
        available = sorted(df["Maturity"].unique())
        raise ValueError(f"No rows found for maturity {maturity}; available maturities are {available}")
    return out.sort_values("Strike").reset_index(drop=True)
