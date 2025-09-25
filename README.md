# FIAM Factor Investing Toolkit

The FIAM repository collects scripts, notebooks, and interim artifacts for building and evaluating equity factor models. The code covers the full pipeline: transforming the raw stock-level panel into monthly partitions, cleaning and scaling factors, selecting a compact signal set, fitting linear and gradient-based models, and back-testing portfolio performance.

## Repository Highlights
- `data/`: raw CSV inputs (returns sample, accounting ratios, link tables, industry data, market factors).
- `parquetization.py`: DuckDB job that rewrites `ret_sample.csv` into year/month partitioned Parquet files.
- `preprocess_parquets.py`: cross-sectional winsorization, return clipping, and schema coercion before modeling.
- `feature_select.py` / `preprocessing`: feature selection pipelines using rank IC filters and gradient-boosted models.
- `penalized_linear_hackathon.py`: expanding-window OLS/Lasso/Ridge/Elastic Net benchmark.
- `portfolio_analysis_hackathon.py`: out-of-sample portfolio sorts, Sharpe ratio, CAPM alpha, turnover, and drawdown diagnostics.

## Data Inputs
The project expects Compustat/CRSP-style monthly security panels with identifiers `gvkey`, `iid`, `excntry`, and `stock_ret` as the one-month ahead return. Important CSV assets shipped in `data/`:
- `ret_sample.csv`: main panel with 147 fundamental/price-based factors and realized returns.
- `factor_char_list.csv`: ordered list of predictors used by the baseline linear model.
- `acc_ratios.csv`: accounting ratio definitions for generating lead ratios (`lead_ratios.py`).
- `mkt_ind.csv`: market benchmark and risk-free series for CAPM-style attribution.
- `cik_gvkey_linktable_USA_only.csv` plus metadata for firm identifiers.

Large derived panels can be stored in `ret_parquets/`, `processed_data/`, or `processed_data2/`. A zipped snapshot (`processed_data.zip`) is included for convenience.

## Environment Setup
1. Install Python 3.10+ and ensure `pip` is available.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
3. Install core packages:
   ```bash
   pip install -r requirements.txt
   ```
   If a requirements file is not yet defined, install manually:
   ```bash
   pip install pandas numpy scipy scikit-learn lightgbm xgboost statsmodels duckdb pyarrow
   ```
4. For GPU-less environments, edit `feature_select.py` (`device="cuda"`) to `device="cpu"` or remove the argument when instantiating `XGBRegressor`.

## Workflow
### 1. Convert CSV to Parquet
```bash
python parquetization.py
```
- Reads `ret_sample.csv` via DuckDB.
- Writes partitioned Parquet files to `ret_parquets/year=YYYY/month=MM/part-0.parquet`.

### 2. Clean Monthly Panels
```bash
python preprocess_parquets.py
```
- Winsorizes heavy-tailed ratios and returns per cross-section.
- Clips extreme returns to [-1, 3].
- Enforces type consistency for IDs and features.
- Outputs cleaned panels under `processed_data2/`.

### 3. Feature Selection
```bash
python feature_select.py
```
- Update `INPUT_ROOT` to your processed Parquet directory (default points to a Kaggle path).
- Pipeline stages: redundancy pruning (`ABS_R_THRESHOLD`), rank IC screen (`rolling_rank_ic`), embedded selection (`XGBRegressor` gain or permutation importance), and optional stability selection.
- Artifacts land in `feature_selection_artifacts/` (kept feature lists, IC table, importances, stability frequencies).

### 4. Baseline Penalized Linear Models
```bash
python penalized_linear_hackathon.py
```
- Set `work_dir` to the folder containing `sample_data.csv` (or point to `ret_sample.csv`).
- Produces expanding-window predictions for OLS, Lasso, Ridge, and Elastic Net.
- Saves combined predictions to `output.csv` and prints out-of-sample R² scores.

### 5. Portfolio Attribution
```bash
python portfolio_analysis_hackathon.py
```
- Point `pred_path` to the model prediction CSV and `mkt_path` to `data/mkt_ind.csv`.
- Forms decile portfolios, computes Sharpe ratio, CAPM alpha with Newey-West errors, maximum drawdown, and long/short turnover.

### 6. Optional Feature Engineering
- `lead_ratios.py`: generates 4-month-ahead accounting ratios and merges them back into the sample.
- `preprocessing` script: alternate feature-selection utilities (rank IC, LightGBM CV, stability selection) for experimentation.

## Notebooks
- `data.ipynb`: exploratory data analysis and sanity checks on the raw panel.
- `gradient_boosting.ipynb`: gradient boosting experiments outside the scripted pipeline.

Run notebooks with Jupyter Lab or VS Code to visualize factor distributions, IC trends, and model diagnostics.

## Directory Layout (abridged)
```
FIAM/
├── data/
├── ret_parquets/
├── processed_data2/
├── feature_selection_artifacts/
├── parquetization.py
├── preprocess_parquets.py
├── feature_select.py
├── penalized_linear_hackathon.py
├── portfolio_analysis_hackathon.py
├── preprocessing
└── README.md
```

## Configuration Notes
- All scripts assume monthly timestamps labeled `year`, `month`, `date`, and realized returns `stock_ret` (next month).
- Keep identifier columns (`gvkey`, `iid`, `excntry`) free of nulls before running selection or modeling.
- Update hard-coded paths (`INPUT_ROOT`, `work_dir`, `pred_path`, `mkt_path`) to match your workspace.
- Set the random seed constants (`SEED`) to reproduce feature rankings.

## Next Steps
- Add `requirements.txt` or `environment.yml` to lock exact package versions.
- Wire a CLI or Makefile to chain preprocessing, selection, training, and evaluation in one command.
- Extend README with data licensing and confidentiality statements when ready to distribute externally.
