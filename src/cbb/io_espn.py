from __future__ import annotations

from pathlib import Path
import pandas as pd
import sportsdataverse.mbb as mbb

from . import config


def _read_parquet(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_all_espn(season: int, refresh: bool = False):
    """
    Load ESPN-derived data via sportsdataverse and cache to data/raw/espn as parquet.

    Returns:
      schedule, pbp, team_box, player_box (all DataFrames)
    """
    raw_dir = config.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    p_schedule = raw_dir / f"schedule_{season}.parquet"
    p_pbp = raw_dir / f"pbp_{season}.parquet"
    p_team_box = raw_dir / f"team_box_{season}.parquet"
    p_player_box = raw_dir / f"player_box_{season}.parquet"

    if not refresh:
        schedule = _read_parquet(p_schedule)
        pbp = _read_parquet(p_pbp)
        team_box = _read_parquet(p_team_box)
        player_box = _read_parquet(p_player_box)
        if all(x is not None for x in [schedule, pbp, team_box, player_box]):
            return schedule, pbp, team_box, player_box

    schedule = mbb.load_mbb_schedule(seasons=[season], return_as_pandas=True)
    pbp = mbb.load_mbb_pbp(seasons=[season], return_as_pandas=True)
    team_box = mbb.load_mbb_team_boxscore(seasons=[season], return_as_pandas=True)
    player_box = mbb.load_mbb_player_boxscore(seasons=[season], return_as_pandas=True)

    _write_parquet(schedule, p_schedule)
    _write_parquet(pbp, p_pbp)
    _write_parquet(team_box, p_team_box)
    _write_parquet(player_box, p_player_box)

    return schedule, pbp, team_box, player_box
