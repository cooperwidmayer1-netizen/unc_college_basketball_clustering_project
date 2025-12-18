from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from .config import EXTERNAL_DIR


# ---- feature groups + weights (your choices) ----
GROUPS = {
    "pace_timing": [
        "poss_pg", "poss_len_sec_mean", "poss_len_sec_p25", "poss_len_sec_p75",
        "trans_shot_rate", "eoc_shot_rate"
    ],
    "off_events": [
        "to_per100", "ft_per100", "foul_per100", "oreb_per100", "dreb_per100"
    ],
    "off_shot_diet": [
        "pct_3", "pct_corner3", "pct_rim", "pct_short_mid", "pct_long_2", "mean_dist_u"
    ],
    "def_shot_diet": [
        "opp_pct_3_allowed", "opp_pct_corner3_allowed", "opp_pct_rim_allowed",
        "opp_pct_short_mid_allowed", "opp_pct_long_2_allowed", "opp_mean_dist_u_allowed"
    ],
}

GROUP_W = {
    "pace_timing": 1.0,
    "off_events": 0.8,
    "off_shot_diet": 1.2,
    "def_shot_diet": 1.2,
}


# ---- name normalization helpers ----
def norm_school(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("&", "and")
    s = s.replace("'", "")
    s = s.replace(".", "")
    s = s.replace("-", " ")
    s = " ".join(s.split())
    return s


def load_aliases() -> dict[str, str]:
    """
    Optional external alias file: data/external/school_aliases.csv
    Columns: from,to
    """
    p = EXTERNAL_DIR / "school_aliases.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    df["from"] = df["from"].astype(str).map(norm_school)
    df["to"] = df["to"].astype(str).map(norm_school)
    return dict(zip(df["from"], df["to"]))


def apply_aliases(s: str, aliases: dict[str, str]) -> str:
    k = norm_school(s)
    return aliases.get(k, k)


def load_kenpom_top100(season: int) -> set[str]:
    """
    Expect file: data/external/kenpom_top100_<season>.txt
    One team per line (e.g. 'North Carolina', 'Iowa St.', etc.)
    """
    p = EXTERNAL_DIR / f"kenpom_top100_{season}.txt"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing KenPom list: {p}\n"
            "Create it with one team per line (Top-100), or change the path logic."
        )
    names = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            names.append(line)
    return set(names)


def zscore_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    scaler = StandardScaler()
    Z = scaler.fit_transform(out[cols].astype(float))
    out[[f"Z_{c}" for c in cols]] = Z
    return out


def _feature_cols(master_style: pd.DataFrame) -> list[str]:
    id_cols = {"team_id", "team_name", "school_name", "school_name_norm", "top_100"}
    return [
        c for c in master_style.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(master_style[c])
    ]


def _weights_for_Z(Z_cols: list[str]) -> np.ndarray:
    w = pd.Series(1.0, index=Z_cols)
    for gname, cols in GROUPS.items():
        gw = float(GROUP_W.get(gname, 1.0))
        for c in cols:
            zc = f"Z_{c}"
            if zc in w.index:
                w[zc] = gw
    return w.to_numpy(dtype=float)


def topk_neighbors(dfZ: pd.DataFrame, Z_cols: list[str], k: int = 3) -> pd.DataFrame:
    dfZ = dfZ.reset_index(drop=True).copy()
    X = dfZ[Z_cols].to_numpy(dtype=float)

    W = _weights_for_Z(Z_cols)
    Xw = X * np.sqrt(W)

    S = cosine_similarity(Xw)

    out_rows = []
    for i in range(len(dfZ)):
        sims = S[i].copy()
        sims[i] = -np.inf
        idx = np.argsort(-sims)[:k]
        for rank, j in enumerate(idx, start=1):
            out_rows.append({
                "team_id": int(dfZ.loc[i, "team_id"]),
                "team_name": dfZ.loc[i, "team_name"],
                "top_100": int(dfZ.loc[i, "top_100"]),
                "neighbor_id": int(dfZ.loc[j, "team_id"]),
                "neighbor_name": dfZ.loc[j, "team_name"],
                "similarity": float(sims[j]),
                "rank": int(rank),
            })
    return pd.DataFrame(out_rows)


def add_kmeans(dfZ: pd.DataFrame, Z_cols: list[str], n_clusters: int = 10, seed: int = 7) -> pd.DataFrame:
    dfZ = dfZ.copy()
    X = dfZ[Z_cols].to_numpy(dtype=float)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    dfZ["cluster"] = km.fit_predict(X)
    return dfZ


def build_neighbors_and_master_z(master_style: pd.DataFrame, season: int):
    """
    Adds:
      - school_name_norm (if school_name exists)
      - top_100 flag based on external kenpom list
      - Z_* columns z-scored within pool (top100 vs non-top100)
      - top-3 cosine neighbors computed within each pool
    """
    ms = master_style.copy()

    # If caller hasn't merged names, fall back to team_id only. We'll try to carry through.
    if "school_name" not in ms.columns:
        ms["school_name"] = ms.get("team_name", ms["team_id"].astype(str))

    aliases = load_aliases()
    ms["school_name_norm"] = ms["school_name"].map(lambda x: apply_aliases(x, aliases))

    kp_raw = load_kenpom_top100(season)
    kp_norm = set(apply_aliases(x, aliases) for x in kp_raw)

    ms["top_100"] = ms["school_name_norm"].isin(kp_norm).astype(int)

    feature_cols = _feature_cols(ms)
    if not feature_cols:
        raise ValueError("No numeric feature columns found in master_style.")

    # pool zscore
    ms_top = ms.loc[ms["top_100"] == 1].copy()
    ms_non = ms.loc[ms["top_100"] == 0].copy()

    ms_top = zscore_df(ms_top, feature_cols) if len(ms_top) else ms_top
    ms_non = zscore_df(ms_non, feature_cols) if len(ms_non) else ms_non

    master_Z = pd.concat([ms_top, ms_non], ignore_index=True)

    Z_cols = [f"Z_{c}" for c in feature_cols]

    # neighbors within pool
    neigh_top = topk_neighbors(master_Z.loc[master_Z["top_100"] == 1], Z_cols, k=3) if len(ms_top) else pd.DataFrame()
    neigh_non = topk_neighbors(master_Z.loc[master_Z["top_100"] == 0], Z_cols, k=3) if len(ms_non) else pd.DataFrame()

    neighbors_top3 = pd.concat([neigh_top, neigh_non], ignore_index=True)
    neighbors_top3 = neighbors_top3.sort_values(["team_name", "rank"]).reset_index(drop=True)

    # clustering (optional; left in master_Z)
    if len(ms_top):
        master_Z.loc[master_Z["top_100"] == 1, "cluster"] = add_kmeans(
            master_Z.loc[master_Z["top_100"] == 1].copy(), Z_cols, n_clusters=10, seed=7
        )["cluster"].to_numpy()
    if len(ms_non):
        master_Z.loc[master_Z["top_100"] == 0, "cluster"] = add_kmeans(
            master_Z.loc[master_Z["top_100"] == 0].copy(), Z_cols, n_clusters=10, seed=7
        )["cluster"].to_numpy()

    return master_Z, neighbors_top3
