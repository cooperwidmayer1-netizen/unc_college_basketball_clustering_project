from __future__ import annotations

import numpy as np
import pandas as pd


def canonicalize_team_display(s: str) -> str:
    s = str(s).strip()
    s = " ".join(s.split())
    return s


def pivot_cards(df_long: pd.DataFrame, team_col: str) -> pd.DataFrame:
    rows = []
    for team, g in df_long.groupby(team_col):
        g = g.sort_values("rank")
        row = {"team": team}
        # keep top_100 if present
        if "top_100" in g.columns and g["top_100"].notna().any():
            row["top_100"] = int(g["top_100"].iloc[0])
        for r in [1, 2, 3]:
            gg = g.loc[g["rank"] == r]
            if len(gg):
                row[f"neighbor_{r}"] = gg["neighbor_name"].iloc[0]
                row[f"sim_{r}"] = float(gg["similarity"].iloc[0])
            else:
                row[f"neighbor_{r}"] = np.nan
                row[f"sim_{r}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)


def infer_acc_conference_id(schedule: pd.DataFrame) -> int:
    acc_like = schedule.loc[
        schedule.get("groups_short_name").astype(str).str.contains(r"\bACC\b", case=False, na=False)
        | schedule.get("groups_name").astype(str).str.contains(r"Atlantic Coast", case=False, na=False)
    ].copy()

    cand_ids = pd.concat(
        [acc_like.get("home_conference_id"), acc_like.get("away_conference_id")],
        ignore_index=True,
    )
    cand_ids = pd.to_numeric(cand_ids, errors="coerce").dropna().astype(int)
    if len(cand_ids) == 0:
        raise ValueError("Could not infer ACC conference_id from schedule.")
    return int(cand_ids.value_counts().index[0])


def build_unc_cards(schedule: pd.DataFrame, neighbors_top3: pd.DataFrame, season: int) -> pd.DataFrame:
    # UNC id inferred from schedule strings
    # We match on home/away_display_name containing "North Carolina"
    unc_games = schedule.loc[
        schedule["home_display_name"].astype(str).str.contains(r"North Carolina", na=False)
        | schedule["away_display_name"].astype(str).str.contains(r"North Carolina", na=False)
    ].copy()

    if len(unc_games) == 0:
        raise ValueError("Could not find UNC games in schedule by display_name containing 'North Carolina'.")

    # Determine UNC team_id from first match
    # Prefer home_id if home is UNC else away_id
    first = unc_games.iloc[0]
    if "North Carolina" in str(first["home_display_name"]):
        unc_id = int(first["home_id"])
    else:
        unc_id = int(first["away_id"])

    unc_games["opponent_id"] = np.where(
        unc_games["home_id"].astype(int) == unc_id,
        unc_games["away_id"].astype(int),
        unc_games["home_id"].astype(int),
    )
    unc_games["opponent_name"] = np.where(
        unc_games["home_id"].astype(int) == unc_id,
        unc_games["away_display_name"],
        unc_games["home_display_name"],
    )

    unc_opps = (
        unc_games[["opponent_id", "opponent_name"]]
        .assign(opponent_name=lambda d: d["opponent_name"].map(canonicalize_team_display))
        .drop_duplicates(subset=["opponent_id"])
        .reset_index(drop=True)
    )

    # join neighbors
    long = unc_opps.merge(
        neighbors_top3[["team_id", "top_100", "neighbor_name", "similarity", "rank"]],
        left_on="opponent_id",
        right_on="team_id",
        how="left",
    ).sort_values(["opponent_name", "rank"])

    wide = pivot_cards(long, team_col="opponent_name")
    return wide


def build_acc_cards(schedule: pd.DataFrame, neighbors_top3: pd.DataFrame, season: int) -> pd.DataFrame:
    acc_id = infer_acc_conference_id(schedule)

    acc_team_ids = pd.concat(
        [
            schedule.loc[pd.to_numeric(schedule["home_conference_id"], errors="coerce").astype("Int64") == acc_id, "home_id"],
            schedule.loc[pd.to_numeric(schedule["away_conference_id"], errors="coerce").astype("Int64") == acc_id, "away_id"],
        ],
        ignore_index=True,
    )
    acc_team_ids = pd.to_numeric(acc_team_ids, errors="coerce").dropna().astype(int).unique().tolist()

    # map ids to names using schedule (most frequent)
    name_rows = []
    for side in [("home_id", "home_display_name"), ("away_id", "away_display_name")]:
        tid, tname = side
        tmp = schedule[[tid, tname]].copy()
        tmp = tmp.rename(columns={tid: "team_id", tname: "team"})
        name_rows.append(tmp)

    name_map = (
        pd.concat(name_rows, ignore_index=True)
        .dropna(subset=["team_id"])
        .assign(team_id=lambda d: pd.to_numeric(d["team_id"], errors="coerce"))
        .dropna(subset=["team_id"])
        .astype({"team_id": "int"})
    )
    name_map["team"] = name_map["team"].astype(str).map(canonicalize_team_display)
    name_map = (
        name_map.value_counts()
        .reset_index(name="n")
        .sort_values(["team_id", "n"], ascending=[True, False])
        .drop_duplicates("team_id")[["team_id", "team"]]
    )

    acc_teams = name_map.loc[name_map["team_id"].isin(acc_team_ids)].copy()

    long = acc_teams.merge(
        neighbors_top3[["team_id", "top_100", "neighbor_name", "similarity", "rank"]],
        on="team_id",
        how="left",
    ).sort_values(["team", "rank"])

    wide = pivot_cards(long, team_col="team")
    return wide