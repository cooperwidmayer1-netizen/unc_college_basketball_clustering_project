"""
One-command pipeline runner.

Usage:
  python scripts/run_pipeline.py --season 2026
  python scripts/run_pipeline.py --season 2026 --refresh-raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cbb import config
from cbb.io_espn import load_all_espn
from cbb.features_style import build_master_style
from cbb.neighbors import build_neighbors_and_master_z
from cbb.cards import build_unc_cards, build_acc_cards
from cbb.reports_pdf import generate_pdfs


def main(season: int, refresh_raw: bool = False) -> None:
    # Ensure directories exist
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.PROC_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[cbb] Season={season} refresh_raw={refresh_raw}")

    # 1) Load ESPN data (cached)
    print("[cbb] Loading ESPN data...")
    schedule, pbp, team_box, player_box = load_all_espn(season=season, refresh=refresh_raw)

    # 2) Build master style table
    print("[cbb] Building master_style...")
    master_style = build_master_style(schedule=schedule, pbp=pbp)

    out_master_style = config.PROC_DIR / f"master_style_{season}.parquet"
    master_style.to_parquet(out_master_style, index=False)
    print(f"[cbb] Wrote {out_master_style}")

    # 3) Build neighbors + master_Z (z-scored pools) + top3 neighbor CSV
    print("[cbb] Building neighbors + master_Z...")
    master_Z, neighbors_top3 = build_neighbors_and_master_z(master_style=master_style, season=season)

    out_master_z = config.PROC_DIR / f"master_Z_{season}.parquet"
    master_Z.to_parquet(out_master_z, index=False)
    print(f"[cbb] Wrote {out_master_z}")

    out_neighbors = config.PROC_DIR / f"espn_style_neighbors_{season}_top3.csv"
    neighbors_top3.to_csv(out_neighbors, index=False)
    print(f"[cbb] Wrote {out_neighbors}")

    # 4) Build UNC opponent cards + ACC cards
    print("[cbb] Building UNC + ACC cards...")
    unc_cards = build_unc_cards(schedule=schedule, neighbors_top3=neighbors_top3, season=season)
    acc_cards = build_acc_cards(schedule=schedule, neighbors_top3=neighbors_top3, season=season)

    out_unc = config.PROC_DIR / f"unc_opponent_cards_top3_{season}.csv"
    out_acc = config.PROC_DIR / f"acc_team_cards_top3_{season}.csv"
    unc_cards.to_csv(out_unc, index=False)
    acc_cards.to_csv(out_acc, index=False)
    print(f"[cbb] Wrote {out_unc}")
    print(f"[cbb] Wrote {out_acc}")

    # 5) Generate PDFs
    print("[cbb] Generating PDFs...")
    generate_pdfs(
        season=season,
        master_Z=master_Z,
        unc_cards_df=unc_cards,
        acc_cards_df=acc_cards,
        outputs_dir=config.OUTPUTS_DIR,
    )

    print("[cbb] DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--refresh-raw", action="store_true")
    args = ap.parse_args()

    main(season=args.season, refresh_raw=args.refresh_raw)
