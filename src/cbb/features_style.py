from __future__ import annotations

import numpy as np
import pandas as pd
from functools import reduce


SHOT_TYPES = ["JumpShot", "LayUpShot", "DunkShot", "TipInShot", "HookShot", "TipShot"]


def clean_coord(s, max_abs=500):
    s = pd.to_numeric(s, errors="coerce")
    return s.mask(s.abs() > max_abs, np.nan)


def parse_clock_sec(series: pd.Series) -> pd.Series:
    clk = series.astype(str)
    mm = pd.to_numeric(clk.str.extract(r"^(?P<m>\d+):")["m"], errors="coerce")
    ss = pd.to_numeric(clk.str.extract(r":(?P<s>\d+)$")["s"], errors="coerce")
    return mm * 60 + ss


def infer_d1_team_ids_from_schedule(schedule: pd.DataFrame, min_games: int = 5) -> set[int]:
    home = schedule[["home_id"]].rename(columns={"home_id": "team_id"})
    away = schedule[["away_id"]].rename(columns={"away_id": "team_id"})

    team_games_id = (
        pd.concat([home, away], ignore_index=True)
        .dropna()
        .value_counts()
        .reset_index(name="games")
    )

    return set(team_games_id.loc[team_games_id["games"] >= min_games, "team_id"].astype(int))


def build_team_shot_style(pbp: pd.DataFrame) -> pd.DataFrame:
    shots = pbp[pbp["type_text"].isin(SHOT_TYPES)].copy()

    shots["x_use"] = clean_coord(shots.get("coordinate_x_raw")).fillna(clean_coord(shots.get("coordinate_x")))
    shots["y_use"] = clean_coord(shots.get("coordinate_y_raw")).fillna(clean_coord(shots.get("coordinate_y")))

    shots = shots.dropna(subset=["x_use", "y_use"]).copy()
    x = shots["x_use"].astype(float)
    y = shots["y_use"].astype(float)

    shots["x0"] = (x - 25.0).abs()
    shots["y0"] = y
    shots["dist_u"] = np.sqrt(shots["x0"] ** 2 + shots["y0"] ** 2)

    shots["is_3"] = shots["text"].astype(str).str.contains(r"three point|3-pt|3pt|3 point", case=False, na=False)

    CORNER_X_U = 22.0
    CORNER_Y_MAX = 16.0
    shots["is_corner3"] = shots["is_3"] & (shots["x0"] >= CORNER_X_U) & (shots["y0"] <= CORNER_Y_MAX)

    RIM_U = 8.0
    SHORT_MID_U = 18.0
    LONG_2_U = 26.0

    def zone_row(r):
        if r["is_3"]:
            return "3pt"
        if r["type_text"] in ("LayUpShot", "DunkShot", "TipInShot"):
            return "rim"
        d = r["dist_u"]
        if d <= RIM_U:
            return "rim"
        if d <= SHORT_MID_U:
            return "short_mid"
        if d <= LONG_2_U:
            return "long_2"
        return "unknown"

    shots["zone"] = shots.apply(zone_row, axis=1)

    team_shot_style = (
        shots.groupby("team_id")
        .agg(
            n_shots=("zone", "size"),
            pct_3=("is_3", "mean"),
            pct_corner3=("is_corner3", "mean"),
            pct_assisted_all=("text", lambda s: s.astype(str).str.contains("assist", case=False).mean()),
            pct_rim=("zone", lambda s: (s == "rim").mean()),
            pct_short_mid=("zone", lambda s: (s == "short_mid").mean()),
            pct_long_2=("zone", lambda s: (s == "long_2").mean()),
            mean_dist_u=("dist_u", "mean"),
        )
        .reset_index()
    )
    return team_shot_style


def build_possessions_and_timing(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    e = pbp.copy()
    e["wallclock_dt"] = pd.to_datetime(e.get("wallclock"), errors="coerce")
    e = e.sort_values(["game_id", "period_number", "wallclock_dt", "id"], kind="mergesort")

    txt = e["text"].astype(str).str.lower()
    e["clock_sec"] = parse_clock_sec(e.get("clock_display_value"))

    e["is_shot"] = e["type_text"].isin(SHOT_TYPES)
    e["is_to"] = e["type_text"].astype(str).str.contains("Turnover", case=False, na=False)
    e["is_ft_made"] = e["type_text"].eq("MadeFreeThrow")
    e["is_end_period"] = e["type_text"].isin(["End Period", "End Game"])

    # keep only rows with team_id
    e_team = e.dropna(subset=["team_id"]).copy()

    e_team["pos_ender"] = e_team["is_shot"] | e_team["is_to"] | e_team["is_ft_made"] | e_team["is_end_period"]
    enders = e_team.loc[e_team["pos_ender"]].copy()

    time_col = None
    for cand in [
        "end_game_seconds_remaining",
        "start_game_seconds_remaining",
        "end_period_seconds_remaining",
        "start_period_seconds_remaining",
    ]:
        if cand in enders.columns:
            time_col = cand
            break
    if time_col is None:
        raise KeyError("No usable *_seconds_remaining column found in pbp.")

    enders["_t"] = pd.to_numeric(enders[time_col], errors="coerce")
    enders = enders.dropna(subset=["_t", "game_id", "team_id"]).copy()
    enders = enders.sort_values(["game_id", "_t"], ascending=[True, False], kind="mergesort")

    enders["_t_next"] = enders.groupby("game_id")["_t"].shift(-1)
    enders["poss_len_sec"] = (enders["_t"] - enders["_t_next"]).clip(lower=0)

    enders["poss_team_id"] = enders["team_id"].astype(int)

    poss_counts = (
        enders.groupby(["game_id", "poss_team_id"])
        .size()
        .reset_index(name="poss")
        .rename(columns={"poss_team_id": "team_id"})
    )

    team_poss_style = (
        poss_counts.groupby("team_id")["poss"]
        .mean()
        .reset_index(name="poss_pg")
        .merge(
            enders.groupby("poss_team_id")
            .agg(
                poss_len_sec_mean=("poss_len_sec", "mean"),
                poss_len_sec_p25=("poss_len_sec", lambda x: np.nanpercentile(x, 25)),
                poss_len_sec_p75=("poss_len_sec", lambda x: np.nanpercentile(x, 75)),
            )
            .reset_index()
            .rename(columns={"poss_team_id": "team_id"}),
            on="team_id",
            how="left",
        )
    )

    timed = enders.copy()
    timed["is_transition_shot"] = timed["poss_len_sec"].notna() & (timed["poss_len_sec"] <= 7) & timed["is_shot"]
    timed["is_eoc_shot"] = timed["poss_len_sec"].notna() & (timed["poss_len_sec"] >= 26) & timed["is_shot"]

    team_timed = (
        timed.groupby("poss_team_id")
        .agg(
            trans_shot_rate=("is_transition_shot", "mean"),
            eoc_shot_rate=("is_eoc_shot", "mean"),
            trans_shots=("is_transition_shot", "sum"),
        )
        .reset_index()
        .rename(columns={"poss_team_id": "team_id"})
    )

    return team_poss_style, team_timed, poss_counts


def build_team_env(pbp: pd.DataFrame, poss_counts: pd.DataFrame) -> pd.DataFrame:
    ev = pbp.copy()
    ev["is_to"] = ev["type_text"].astype(str).str.contains("Turnover", case=False, na=False)
    ev["is_ft"] = ev["type_text"].isin(["MadeFreeThrow", "MissedFreeThrow"])
    ev["is_foul"] = ev["type_text"].astype(str).str.contains("Foul", case=False, na=False)
    ev["is_oreb"] = ev["text"].astype(str).str.contains("offensive rebound", case=False, na=False)
    ev["is_dreb"] = ev["text"].astype(str).str.contains("defensive rebound", case=False, na=False)

    team_event_counts = (
        ev.groupby("team_id")
        .agg(
            to_cnt=("is_to", "sum"),
            ft_cnt=("is_ft", "sum"),
            foul_cnt=("is_foul", "sum"),
            oreb_cnt=("is_oreb", "sum"),
            dreb_cnt=("is_dreb", "sum"),
        )
        .reset_index()
    )

    team_poss_tot = poss_counts.groupby("team_id")["poss"].sum().reset_index(name="poss_tot")

    team_env = team_event_counts.merge(team_poss_tot, on="team_id", how="left")

    for c in ["to", "ft", "foul", "oreb", "dreb"]:
        team_env[f"{c}_per100"] = team_env[f"{c}_cnt"] / team_env["poss_tot"] * 100

    return team_env[["team_id", "to_per100", "ft_per100", "foul_per100", "oreb_per100", "dreb_per100"]]


def build_defensive_shot_diet(pbp: pd.DataFrame) -> pd.DataFrame:
    shots_def = pbp[pbp["type_text"].isin(SHOT_TYPES)].copy()

    shots_def["off_team_id"] = shots_def["team_id"]

    shots_def["def_team_id"] = np.where(
        shots_def["off_team_id"] == shots_def["home_team_id"],
        shots_def["away_team_id"],
        shots_def["home_team_id"],
    )

    shots_def = shots_def.dropna(subset=["def_team_id", "off_team_id"]).copy()
    shots_def["def_team_id"] = shots_def["def_team_id"].astype(int)

    x = pd.to_numeric(shots_def.get("coordinate_x"), errors="coerce")
    y = pd.to_numeric(shots_def.get("coordinate_y"), errors="coerce")

    if "coordinate_x_raw" in shots_def.columns and "coordinate_y_raw" in shots_def.columns:
        xr = pd.to_numeric(shots_def.get("coordinate_x_raw"), errors="coerce")
        yr = pd.to_numeric(shots_def.get("coordinate_y_raw"), errors="coerce")
        x = x.where(x.notna(), xr)
        y = y.where(y.notna(), yr)

    shots_def["x"] = x
    shots_def["y"] = y
    shots_def = shots_def.dropna(subset=["x", "y"]).copy()

    m = np.maximum(shots_def["x"].abs(), shots_def["y"].abs()).to_numpy()
    scale_pow = np.zeros(len(shots_def), dtype=int)
    while True:
        mask = m > 120
        if not mask.any():
            break
        m[mask] = m[mask] / 10.0
        scale_pow[mask] += 1
    scale = (10.0 ** scale_pow)
    shots_def["x"] = shots_def["x"] / scale
    shots_def["y"] = shots_def["y"] / scale

    shots_def["dist_u"] = np.sqrt(shots_def["x"] ** 2 + shots_def["y"] ** 2)
    shots_def["is_3"] = shots_def.get("points_attempted").eq(3)

    shots_def["is_corner3"] = shots_def["is_3"] & (np.abs(shots_def["x"]) >= 22) & (shots_def["y"] <= 5)

    # Empirical cuts from your script
    RIM_CUT = 24.20872776500244
    SHORT_MID_CUT = 36.75

    def classify_zone(row):
        if row["is_3"]:
            return "3"
        if row["dist_u"] <= RIM_CUT:
            return "rim"
        if row["dist_u"] <= SHORT_MID_CUT:
            return "short_mid"
        return "long_2"

    shots_def["zone"] = shots_def.apply(classify_zone, axis=1)

    def_shot_diet = (
        shots_def.groupby("def_team_id")
        .agg(
            opp_n_shots=("zone", "size"),
            opp_pct_3_allowed=("is_3", "mean"),
            opp_pct_corner3_allowed=("is_corner3", "mean"),
            opp_pct_rim_allowed=("zone", lambda s: (s == "rim").mean()),
            opp_pct_short_mid_allowed=("zone", lambda s: (s == "short_mid").mean()),
            opp_pct_long_2_allowed=("zone", lambda s: (s == "long_2").mean()),
            opp_mean_dist_u_allowed=("dist_u", "mean"),
        )
        .reset_index()
        .rename(columns={"def_team_id": "team_id"})
    )

    return def_shot_diet


def build_master_style(schedule: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    d1_ids = infer_d1_team_ids_from_schedule(schedule)

    team_poss_style, team_timed, poss_counts = build_possessions_and_timing(pbp)
    team_env = build_team_env(pbp, poss_counts)
    team_shot_style = build_team_shot_style(pbp)
    def_shot_diet = build_defensive_shot_diet(pbp)

    tables = [team_poss_style, team_timed, team_env, team_shot_style, def_shot_diet]
    master_style = reduce(lambda left, right: left.merge(right, on="team_id", how="inner"), tables)

    # Add team_name from schedule (most frequent display name per team_id)
    name_rows = []
    for tid, tname in [("home_id","home_display_name"), ("away_id","away_display_name")]:
        if tid in schedule.columns and tname in schedule.columns:
            tmp = schedule[[tid, tname]].copy()
            tmp = tmp.rename(columns={tid: "team_id", tname: "team_name"})
            name_rows.append(tmp)
    if name_rows:
        name_map = (
            pd.concat(name_rows, ignore_index=True)
            .dropna(subset=["team_id"])
            .assign(team_id=lambda d: pd.to_numeric(d["team_id"], errors="coerce"))
            .dropna(subset=["team_id"])
            .astype({"team_id": "int"})
        )
        name_map["team_name"] = name_map["team_name"].astype(str).str.strip()
        name_map = (
            name_map.value_counts()
            .reset_index(name="n")
            .sort_values(["team_id", "n"], ascending=[True, False])
            .drop_duplicates("team_id")[["team_id", "team_name"]]
        )
        master_style = master_style.merge(name_map, on="team_id", how="left")

    master_style = master_style[master_style["team_id"].isin(d1_ids)].copy()
    return master_style
