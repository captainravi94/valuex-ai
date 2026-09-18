import io
import re
import requests
import openpyxl
import pandas as pd
import numpy as np

def _parse_screener_datasheet(wb, uploaded_file):
    """
    Parses canonical Screener.in workbooks directly from 'Data Sheet',
    handling companies with partial historical periods (e.g., Hyundai's 6-yr vs Vodafone's 10-yr).
    """
    ws = wb['Data Sheet']
    
    # 1. Company Name from cell B1
    comp_name = ws.cell(1, 2).value
    if not comp_name:
        raw_name = uploaded_file.name.rsplit(".", 1)[0]
        comp_name = re.sub(r"[_\-\d]+", " ", raw_name).strip().title()
    else:
        comp_name = str(comp_name).strip().title()

    # 2. Extract Active Date Columns from Row 16 (Report Date under PROFIT & LOSS)
    # Hyundai has blank columns B-E and starts at F (FY21); Vodafone starts at B (FY17).
    date_cols = {}
    for col_idx in range(2, ws.max_column + 1):
        dt_val = ws.cell(16, col_idx).value
        if dt_val is not None:
            if hasattr(dt_val, "strftime"):
                label = dt_val.strftime("FY%y")
            else:
                label = str(dt_val).strip()
            date_cols[col_idx] = label

    if not date_cols:
        return None

    # 3. Harvest metrics across P&L, Balance Sheet, and Cash Flow blocks
    lines = {}
    total_count = 0
    for row_idx in range(16, ws.max_row + 1):
        metric_cell = ws.cell(row_idx, 1).value
        if not metric_cell:
            continue
        m_str = str(metric_cell).strip()
        
        # Skip section titles and repeated date headers
        if m_str.upper() in ["PROFIT & LOSS", "QUARTERS", "BALANCE SHEET", "CASH FLOW:", "REPORT DATE"]:
            continue

        row_vals = {}
        for c_idx, period_label in date_cols.items():
            cell_v = ws.cell(row_idx, c_idx).value
            try:
                row_vals[period_label] = float(cell_v) if cell_v is not None else 0.0
            except (ValueError, TypeError):
                row_vals[period_label] = 0.0

        # Handle duplicate line-item names (e.g., 'Total' in BS liabilities vs assets)
        if m_str.lower() == "total":
            total_count += 1
            metric_key = "Total Liabilities" if total_count == 1 else "total_assets"
        elif m_str.lower() in [k.lower() for k in lines]:
            metric_key = f"{m_str}_{row_idx}"
        else:
            metric_key = m_str

        lines[metric_key] = row_vals

    # Construct clean DataFrame
    df = pd.DataFrame(lines).T
    df.index.name = "metric"
    df.columns = [str(c) for c in df.columns]

    # Map Operating Cash Flow if Screener labeled it 'Cash from Operating Activity'
    for idx in list(df.index):
        if "operating activit" in str(idx).lower() and "cfo" not in [i.lower() for i in df.index]:
            df.loc["cfo"] = df.loc[idx]
        if "borrowings" in str(idx).lower() and "total_borrowings" not in [i.lower() for i in df.index]:
            df.loc["total_borrowings"] = df.loc[idx]
        if "cash & bank" in str(idx).lower() and "cash_and_equivalents" not in [i.lower() for i in df.index]:
            df.loc["cash_and_equivalents"] = df.loc[idx]

    # Synthetic EBITDA calculation if not isolated in standard Screener sheet
    if "ebitda" not in [str(i).lower() for i in df.index]:
        sales_key = next((k for k in df.index if k.lower() == "sales"), None)
        pbt_key = next((k for k in df.index if "profit before tax" in k.lower()), None)
        dep_key = next((k for k in df.index if "depreciation" in k.lower()), None)
        int_key = next((k for k in df.index if "interest" in k.lower()), None)
        
        if pbt_key and dep_key and int_key:
            df.loc["ebitda"] = df.loc[pbt_key] + df.loc[dep_key] + df.loc[int_key]
        elif sales_key:
            exp_rows = [r for r in df.index if any(x in r.lower() for x in ["raw material", "employee", "power", "selling", "other mfr", "other expenses"])]
            if exp_rows:
                df.loc["ebitda"] = df.loc[sales_key] - df.loc[exp_rows].sum()

    currency = "$" if any(w in comp_name.lower() for w in ["inc", "corp", "apple", "tesla", "citi"]) else "₹"
    scale = "Cr" if currency == "₹" else "M"

    return df, f"Successfully parsed {len(df.columns)} financial periods for {comp_name}", comp_name, "Corporate / Industrial", currency, scale, {"periods": list(df.columns)}


def ingest_from_excel_or_csv(uploaded_file):
    try:
        fname = uploaded_file.name.lower()
        
        # --- PATH A: Dedicated Screener.in Ingestion Engine ---
        if fname.endswith((".xlsx", ".xls")):
            uploaded_file.seek(0)
            wb = openpyxl.load_workbook(uploaded_file, data_only=True)
            if "Data Sheet" in wb.sheetnames:
                result = _parse_screener_datasheet(wb, uploaded_file)
                if result is not None:
                    return result

        # --- PATH B: Standard Tabular CSV / Non-Screener Excel Ingestion ---
        uploaded_file.seek(0)
        if fname.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file, header=None)
        else:
            xl = pd.ExcelFile(uploaded_file)
            sheet_to_use = xl.sheet_names[0]
            for s in xl.sheet_names:
                if any(k in s.lower() for k in ["profit", "consolidated", "standalone", "sheet1"]):
                    sheet_to_use = s
                    break
            df_raw = pd.read_excel(xl, sheet_name=sheet_to_use, header=None)

        # Hunt for header row containing fiscal period labels
        header_row_idx = None
        for idx, row in df_raw.iloc[:50].iterrows():
            row_items = [str(v).strip() for v in row.values if pd.notna(v)]
            matches = [v for v in row_items if re.search(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|fy|\b20\d\d\b)", v, re.IGNORECASE)]
            if len(matches) >= 2:
                header_row_idx = idx
                break

        if header_row_idx is None:
            header_row_idx = 0

        raw_headers = list(df_raw.iloc[header_row_idx].values)
        headers = []
        for h in raw_headers:
            if hasattr(h, "strftime"):
                headers.append(h.strftime("%b %Y"))
            elif pd.isna(h):
                headers.append("")
            else:
                headers.append(str(h).replace("\n", " ").strip())

        df = df_raw.iloc[header_row_idx + 1:].copy()
        
        # Deduplicate column headers
        seen = {}
        deduped_headers = []
        for h in headers:
            if h not in seen:
                seen[h] = 0
                deduped_headers.append(h)
            else:
                seen[h] += 1
                deduped_headers.append(f"{h}_{seen[h]}")
        df.columns = deduped_headers

        first_col = df.columns[0]
        df = df.rename(columns={first_col: "metric"})
        df = df[df["metric"].notna()]
        df["metric"] = df["metric"].astype(str).str.strip()
        df = df[~df["metric"].str.lower().str.contains("screener|http|www|report|source|notes", regex=True)]
        df = df[df["metric"] != ""]

        # Keep columns containing numeric data
        valid_cols = ["metric"]
        for col in df.columns[1:]:
            s = df[col].iloc[:, 0] if isinstance(df[col], pd.DataFrame) else df[col]
            num_count = pd.to_numeric(
                s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip(),
                errors="coerce"
            ).notna().sum()
            if num_count >= 2:
                valid_cols.append(col)

        df = df[valid_cols]

        period_cols = [c for c in df.columns if c != "metric"]
        for col in period_cols:
            s = df[col].iloc[:, 0] if isinstance(df[col], pd.DataFrame) else df[col]
            clean_s = (
                s.astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("(", "-", regex=False)
                .str.replace(")", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(clean_s, errors="coerce").fillna(0.0)

        df = df.set_index("metric")
        df = df[~df.index.duplicated(keep="first")]

        raw_name = uploaded_file.name.rsplit(".", 1)[0]
        comp_name = re.sub(r"[_\-\d]+", " ", raw_name).strip().title()
        currency = "$" if any(w in comp_name.lower() for w in ["inc", "corp", "apple", "tesla", "citi"]) else "₹"
        scale = "Cr" if currency == "₹" else "M"

        return df, f"Successfully parsed {len(df.columns)} financial periods for {comp_name}", comp_name, "Corporate / Industrial", currency, scale, {"periods": list(df.columns)}

    except Exception as e:
        return None, f"Ingestion error: {str(e)}", None, None, "₹", "Cr", {}

def fetch_from_ir_link(url: str):
    try:
        if not url.startswith("http"):
            url = "https://" + url
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(io.StringIO(resp.text))
        if tables:
            best = max(tables, key=lambda t: t.shape[0] * t.shape[1])
            return best, f"Extracted table ({best.shape[0]} rows x {best.shape[1]} cols)."
        return None, "No HTML tables detected"
    except Exception as e:
        return None, str(e)