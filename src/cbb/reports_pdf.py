from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER


FEATURE_GROUPS = {
    "Pace & Timing": [
        "poss_pg", "poss_len_sec_mean", "poss_len_sec_p25", "poss_len_sec_p75",
        "trans_shot_rate", "eoc_shot_rate"
    ],
    "Offensive Events": [
        "to_per100", "ft_per100", "foul_per100", "oreb_per100", "dreb_per100"
    ],
    "Offensive Shot Profile": [
        "pct_3", "pct_corner3", "pct_rim", "pct_short_mid", "pct_long_2", "mean_dist_u"
    ],
    "Defensive Shot Profile": [
        "opp_pct_3_allowed", "opp_pct_corner3_allowed", "opp_pct_rim_allowed",
        "opp_pct_short_mid_allowed", "opp_pct_long_2_allowed", "opp_mean_dist_u_allowed"
    ],
}

REQUIRED_CARD_COLS = ["team", "neighbor_1", "neighbor_2", "neighbor_3", "sim_1", "sim_2", "sim_3"]


def _coerce_cards_df(cards_df: pd.DataFrame) -> pd.DataFrame:
    df = cards_df.copy()
    missing = [c for c in REQUIRED_CARD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"cards_df missing required columns: {missing}")
    df["team"] = df["team"].astype(str)
    for c in ["sim_1", "sim_2", "sim_3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "top_100" in df.columns:
        df["top_100"] = pd.to_numeric(df["top_100"], errors="coerce").fillna(0).astype(int)
    return df


def _build_master_index(master_Z: pd.DataFrame) -> Dict[str, int]:
    if "team_name" not in master_Z.columns:
        # If master_style didn't have a team_name, fall back to team_id str
        raise ValueError("master_Z must include a 'team_name' column for PDF lookup.")
    name_to_idx = {}
    for idx, name in zip(master_Z.index, master_Z["team_name"].astype(str)):
        if name not in name_to_idx:
            name_to_idx[name] = idx
    return name_to_idx


def _category_avgs(team_row: pd.Series) -> dict:
    out = {}
    for group_name, feats in FEATURE_GROUPS.items():
        zs = []
        for f in feats:
            zc = f"Z_{f}"
            if zc in team_row.index and pd.notna(team_row[zc]):
                zs.append(float(team_row[zc]))
        out[group_name] = float(np.mean(zs)) if zs else np.nan
    return out


def _fmt_sim(x) -> str:
    if pd.isna(x):
        return "—"
    x = float(x)
    return f"{x:.3f} ({x*100:.1f}%)"


def generate_team_card_pdf(
    cards_df: pd.DataFrame,
    master_Z: pd.DataFrame,
    output_path: Path,
    title_prefix: str,
):
    cards_df = _coerce_cards_df(cards_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    master_index = _build_master_index(master_Z)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=20,
        spaceAfter=4,
        textColor=colors.HexColor("#13294B"),
        alignment=TA_CENTER,
    )
    header_style = ParagraphStyle(
        "Header",
        parent=styles["Normal"],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#666666"),
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#555555"),
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#13294B"),
        fontName="Helvetica-Bold",
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title=f"{title_prefix} Style Cards",
    )

    story = []
    for i, row in cards_df.iterrows():
        team = row["team"]

        story.append(Paragraph(team, title_style))
        story.append(Paragraph(f"{title_prefix} Style Card", header_style))
        story.append(Spacer(1, 0.10 * inch))

        story.append(Paragraph("Most Similar Teams", h2))
        neighbor_data = [["#", "Team", "Similarity"]]
        for r in [1, 2, 3]:
            neighbor_data.append([str(r), str(row.get(f"neighbor_{r}", "—")), _fmt_sim(row.get(f"sim_{r}"))])

        t = Table(neighbor_data, colWidths=[0.4 * inch, 4.6 * inch, 1.3 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#13294B")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 0.12 * inch))

        # category overview (table only, no chart)
        story.append(Paragraph("Category Overview (avg Z within pool)", h2))

        idx = master_index.get(team)
        if idx is None:
            story.append(
                Paragraph(
                    "<b>No master_Z row found for this team name.</b><br/>"
                    "This is usually a naming mismatch between cards_df.team and master_Z.team_name.",
                    small_style,
                )
            )
        else:
            team_row = master_Z.loc[idx]
            avgs = _category_avgs(team_row)
            data = [["Category", "Avg Z"]]
            for cat, v in avgs.items():
                data.append([cat, "—" if pd.isna(v) else f"{float(v):+.2f}"])
            t2 = Table(data, colWidths=[4.6 * inch, 1.7 * inch])
            t2.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A90E2")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("ALIGN", (1, 1), (1, -1), "CENTER"),
                    ]
                )
            )
            story.append(t2)

        story.append(Spacer(1, 0.10 * inch))
        story.append(Paragraph("<i>Z-scores describe style/volume (not “good/bad”).</i>", small_style))

        if i < len(cards_df) - 1:
            story.append(PageBreak())

    doc.build(story)


def generate_pdfs(
    season: int,
    master_Z: pd.DataFrame,
    unc_cards_df: pd.DataFrame,
    acc_cards_df: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    generate_team_card_pdf(
        cards_df=unc_cards_df,
        master_Z=master_Z,
        output_path=outputs_dir / f"unc_opponents_style_cards_{season}.pdf",
        title_prefix="UNC OPPONENT",
    )
    generate_team_card_pdf(
        cards_df=acc_cards_df,
        master_Z=master_Z,
        output_path=outputs_dir / f"acc_teams_style_cards_{season}.pdf",
        title_prefix="ACC TEAM",
    )