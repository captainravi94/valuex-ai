import io
import re
import pandas as pd
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def sanitize_markdown_for_reportlab(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = raw_text.replace(r"$\rightarrow$", "→").replace(r"\rightarrow", "→")
    text = text.replace("👉", "→").replace("⚠️", "[!]").replace("🔵", "[Reported]").replace("🟣", "[Calculated]").replace("🟠", "[Observation]").replace("🔴", "[Action]")
    text = re.sub(r"^\s*[\*\-]\s+", "• ", text)

    def clean_code(match):
        c_text = match.group(1)
        c_text = c_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("*", "×")
        return f"<font face='Courier'>{c_text}</font>"

    text = re.sub(r"`(.+?)`", clean_code, text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("&lt;font face='Courier'&gt;", "<font face='Courier'>").replace("&lt;/font&gt;", "</font>")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\b\*([^\*\n]+?)\*\b", r"<i>\1</i>", text)
    text = text.replace("*", "•")
    text = re.sub(r"^#+\s*", "", text)

    return text.strip()

class InstitutionalPDFExporter:
    @staticmethod
    def build_pdf_dossier(
        company_name: str,
        period: str,
        taxonomy: str,
        unit: str,
        scale: str,
        recon: Dict[str, Any],
        df_metrics: pd.DataFrame,
        df_ratios: pd.DataFrame,
        altman: Dict[str, Any],
        piotroski: Dict[str, Any],
        anomalies: List[Dict[str, Any]],
        dossier_text: str
    ) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        c_primary = colors.HexColor("#0f172a")
        c_secondary = colors.HexColor("#1e293b")
        c_accent = colors.HexColor("#2563eb")
        c_border = colors.HexColor("#cbd5e1")
        c_bg_light = colors.HexColor("#f8fafc")

        title_style = ParagraphStyle(
            "DocTitle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=c_primary
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=9, leading=12, textColor=colors.HexColor("#64748b")
        )
        h1_style = ParagraphStyle(
            "Heading1_Custom", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=c_secondary, spaceBefore=8, spaceAfter=4
        )
        body_style = ParagraphStyle(
            "Body_Custom", parent=styles["Normal"], fontName="Helvetica", fontSize=7.8, leading=11, textColor=colors.HexColor("#334155")
        )
        bold_body = ParagraphStyle(
            "BoldBody_Custom", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7.8, leading=11, textColor=c_primary
        )

        story = []

        # Header Block
        story.append(Paragraph("VALUEX AI | EXECUTIVE FINANCIAL DOSSIER", subtitle_style))
        story.append(Spacer(1, 3))
        story.append(Paragraph(sanitize_markdown_for_reportlab(company_name.upper()), title_style))

        meta_line = (
            f"<b>Taxonomy:</b> {sanitize_markdown_for_reportlab(taxonomy)} | "
            f"<b>Valuation Cycle:</b> {period} | "
            f"<b>Scale:</b> {unit} in {scale} | "
            f"<b>Data Confidence:</b> {recon.get('confidence_pct', 0)}% ({recon.get('overall_status', 'N/A')})"
        )
        story.append(Paragraph(meta_line, body_style))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=0, spaceAfter=8))

        # 1. Statement Quality & Reconciliation Scorecard
        story.append(Paragraph("1. FINANCIAL STATEMENT RECONCILIATION & DATA QUALITY", h1_style))
        recon_rows = [["Audit Test", "Status", "Observed Evidence / Disclosures"]]
        for chk in recon.get("checks", [])[:6]:
            recon_rows.append([
                Paragraph(sanitize_markdown_for_reportlab(chk.get("test", "")), bold_body),
                Paragraph(f"<b>{chk.get('status', '')}</b>", body_style),
                Paragraph(sanitize_markdown_for_reportlab(chk.get("evidence", "")), body_style)
            ])

        t_recon = Table(recon_rows, colWidths=[150, 75, 315])
        t_recon.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_secondary),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("GRID", (0, 0), (-1, -1), 0.5, c_border),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_bg_light])
        ]))
        story.append(t_recon)
        story.append(Spacer(1, 8))

        # 2. Audited Statements (Spreaded)
        story.append(Paragraph("2. AUDITED FINANCIAL STATEMENTS (SPREADED)", h1_style))
        p_cols = [c for c in df_metrics.columns if str(c).lower() not in ["metric", "category", "line_item"]]
        table_cols = ["Line Item"] + p_cols[-4:]

        spread_rows = [table_cols]
        for idx in df_metrics.index[:9]:
            r_vals = [Paragraph(str(idx).replace("_", " ").title(), bold_body)]
            for col in p_cols[-4:]:
                val = df_metrics.loc[idx, col]
                try:
                    num = float(val)
                    r_vals.append(f"{unit}{num:,.0f}" if abs(num) >= 1000 else f"{unit}{num:,.2f}")
                except Exception:
                    r_vals.append(str(val))
            spread_rows.append(r_vals)

        c_w = 390 / max(len(p_cols[-4:]), 1)
        t_spread = Table(spread_rows, colWidths=[150] + [c_w] * len(p_cols[-4:]))
        t_spread.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_primary),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.5, c_border),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_bg_light])
        ]))
        story.append(t_spread)
        story.append(Spacer(1, 8))

        # 3. Detected Anomalies
        if anomalies:
            story.append(Paragraph("3. DETECTED FINANCIAL ANOMALIES & AUDIT DIVERGENCES", h1_style))
            anom_rows = [["ID", "Severity", "Detected Divergence", "Investigation Directive"]]
            for an in anomalies[:4]:
                anom_rows.append([
                    Paragraph(sanitize_markdown_for_reportlab(an.get("id", "")), bold_body),
                    Paragraph(f"<b>{an.get('severity', '')}</b>", body_style),
                    Paragraph(f"<b>{sanitize_markdown_for_reportlab(an.get('title', ''))}</b><br/>{sanitize_markdown_for_reportlab(an.get('evidence', ''))}", body_style),
                    Paragraph(sanitize_markdown_for_reportlab(an.get("investigation", "")), body_style)
                ])
            t_anom = Table(anom_rows, colWidths=[60, 50, 215, 215])
            t_anom.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7f1d1d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.5, c_border),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_bg_light])
            ]))
            story.append(t_anom)
            story.append(Spacer(1, 8))

        story.append(PageBreak())

        # 4. Seven-Tier Qualitative Findings
        story.append(Paragraph("4. SEVEN-TIER FORENSIC AUDIT & QUALITATIVE REASONING", h1_style))
        raw_lines = dossier_text.split("\n")
        for raw_line in raw_lines:
            line_str = raw_line.strip()
            if not line_str:
                continue

            if line_str == "---":
                story.append(HRFlowable(width="100%", thickness=0.5, color=c_border, spaceBefore=3, spaceAfter=4))
                continue

            if line_str.startswith("###") or line_str.startswith("##"):
                clean_header = sanitize_markdown_for_reportlab(line_str)
                story.append(Paragraph(clean_header, h1_style))
                continue

            sanitized_line = sanitize_markdown_for_reportlab(line_str)
            if sanitized_line:
                story.append(Paragraph(sanitized_line, body_style))
                story.append(Spacer(1, 1.5))

        story.append(Spacer(1, 8))

        # 5. Model Validation Summary
        story.append(Paragraph("5. SOLVENCY & QUALITY MODEL METHODOLOGY AUDIT", h1_style))
        model_rows = [
            ["Model", "Calculated Score", "Classification", "Applicability / Inputs"],
            [
                Paragraph("<b>Altman Z-Score</b>", bold_body),
                str(altman.get("z_score", "N/A")),
                altman.get("zone", "N/A"),
                Paragraph(sanitize_markdown_for_reportlab(altman.get("formula_version", "")), body_style)
            ],
            [
                Paragraph("<b>Piotroski F-Score</b>", bold_body),
                f"{piotroski.get('passing_count', 0)} Passes",
                piotroski.get("summary_label", "N/A"),
                Paragraph(f"Verifiable: {piotroski.get('verifiable_count', 0)}/9 | Unavailable: {piotroski.get('unavailable_count', 0)}", body_style)
            ]
        ]
        t_model = Table(model_rows, colWidths=[110, 80, 150, 200])
        t_model.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_secondary),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.5, c_border),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_bg_light])
        ]))
        story.append(t_model)

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()