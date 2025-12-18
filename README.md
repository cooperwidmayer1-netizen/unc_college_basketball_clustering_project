UNC College Basketball Style Clustering

This project builds team “style” profiles from ESPN men’s college basketball play-by-play (via sportsdataverse), finds nearest-neighbor style comps using weighted cosine similarity, clusters teams, and generates PDF “style cards” for UNC opponents and ACC teams.

What it does

Pull ESPN schedule + play-by-play + boxscores for a season

Engineer team style features (pace/timing, events, shot diet, defensive shot diet)

Mark Top-100 teams from a KenPom-style list you provide

Z-score features within Top-100 and within Non-Top-100 pools

Compute top-3 nearest neighbors (cosine similarity) + KMeans clusters

Produce CSVs and PDF reports

Repo layout (high level)

src/cbb/ core library code

scripts/run_pipeline.py one command runs everything

data/external/ inputs you provide (KenPom top-100 list)

data/raw/ cached ESPN pulls (usually not committed)

data/processed/ generated artifacts (usually not committed)

outputs/ generated PDFs (optional to commit)

Requirements

Python 3.9+

macOS: Homebrew recommended

brew install xz libomp

Setup (macOS)

brew install xz libomp

python3 -m venv .venv

source .venv/bin/activate

python -m pip install -U pip setuptools wheel

python -m pip install -e .

python -m pip install "xgboost<3.1"

Input: Top-100 list
Create this file:
data/external/kenpom_top100_2026.txt

Format: one team per line (100 lines total). Example lines:
North Carolina
Duke
Kansas
Connecticut
...

Run
source .venv/bin/activate
python scripts/run_pipeline.py --season 2026

Outputs (example for 2026)
data/processed/master_style_2026.parquet
data/processed/master_Z_2026.parquet
data/processed/espn_style_neighbors_2026_top3.csv
data/processed/unc_opponent_cards_top3_2026.csv
data/processed/acc_team_cards_top3_2026.csv
outputs/unc_opponents_style_cards_2026.pdf
outputs/acc_teams_style_cards_2026.pdf

Notes / common issues

If you see an OpenMP / libomp.dylib error: brew install libomp

If you see lzma.h not found: brew install xz

sportsdataverse may import CFB modules internally; this can trigger xgboost dependency checks even if you only use MBB
