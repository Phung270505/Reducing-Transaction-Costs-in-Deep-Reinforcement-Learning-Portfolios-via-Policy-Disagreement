# Reducing Transaction Costs in Deep Reinforcement Learning Portfolios via Policy Disagreement

Research code, the frozen input dataset, and stored results for DEFI 2026
submission 73, by Van-Phung Nguyen, Ba-Nhan Le, and Lam Mai.

DAR-Soft controls execution after inference. It blends the arithmetic mean
of five independently trained PPO policies with the current holdings. The
adjustment coefficient depends on disagreement between their proposed
allocations. The policies remain frozen during execution-control calibration
and evaluation.

## Research design

For cash-inclusive proposals `w_i`, DAR-Soft uses:

```text
c = mean_i(w_i)
D_c = mean_i(0.5 * ||w_i - c||_1)
alpha_t = exp(-k * D_c)
w_target = (1 - alpha_t) * w_current + alpha_t * c
```

Initial deployment uses `alpha_0 = 1`; subsequent decisions use `k = 0.5`,
selected on the 2023 calibration period. Current holdings are marked at the
preceding close. Realized holdings can differ from targets because of lot,
cash, liquidity, and settlement constraints.

The study uses a fixed, ex-post basket of 30 Vietnamese equities plus cash.
VN-Index supplies market context and a price-return benchmark. The PPO
network uses recurrent stock/market encoders, temporal attention, and an
asset-level Transformer. The full model, reward, execution, and calibration
settings are stored in `artifacts/disagreement_ppo_main/experiment_config.json`.

| Component | Setting |
| --- | --- |
| PPO training | 2016-2021 |
| Checkpoint validation | 2022 |
| Execution-control calibration | 2023 |
| Retrospective development evaluation | 2024-2025 |
| Independent PPO seeds | 5, 10, 15, 20, 25 |
| Training budget | 200,000 decision transitions per seed |
| Observation lookback / rebalancing | 20 market sessions / every 20 market sessions |
| Evaluation sample | 499 daily observations, 25 decision dates, 24 complete forward windows |
| Initial capital | 1 billion VND |
| Modeled costs | 0.15% brokerage on buys/sells, 0.10% sell tax, 5 bps slippage |
| Settlement | Two-session approximation |

The 2024-2025 period informed development and is not an untouched confirmatory
holdout. Historical configuration keys named `test` do not change that scope.

## Main stored result

| Strategy | CAGR (%) | Sharpe | Annual turnover | Cost (million VND) |
| --- | ---: | ---: | ---: | ---: |
| PPO mean | 49.16 | 2.562 | 5.662 | 39.37 |
| DAR-Soft | 49.16 | 2.544 | 5.308 | 36.85 |
| Fixed-alpha control | 49.35 | 2.549 | 5.306 | 36.89 |
| Retrospective shuffled timing | 49.10 | 2.539 | 5.319 | 36.91 |

DAR-Soft reduces annualized realized turnover by approximately 6.3% and
modeled transaction costs by 6.41% relative to ungated PPO mean. Static
controls and the shuffled schedule achieve comparable outcomes. The stored
confidence intervals do not establish a distinct CAGR or Sharpe advantage
attributable to disagreement timing.

## Source files

Use the current project directory as the Git repository root. `.gitignore`
contains an explicit selection of the original files needed for this study.

| Path | Role |
| --- | --- |
| `main.py` | Original data-inspection, training, and evaluation CLI |
| `train_ppo.py` | Train the five PPO checkpoints |
| `optimize_ensemble.py` | Calibrate and evaluate execution controls from checkpoints |
| `reviewer_evaluation.py` | Generic controls, statistical comparisons, diagnostics, and plots |
| `vn30ppo/config.py`, `data.py` | Experiment settings, features, scaling, and asset availability |
| `vn30ppo/environment.py` | Portfolio simulator, accounting, and execution constraints |
| `vn30ppo/ppo.py` | PPO network, training, checkpoint selection, and inference |
| `vn30ppo/evaluation.py`, `experiment.py` | Execution controls, calibration, backtests, and metrics |
| `vn30ppo/research_protocol.py`, `reviewer_statistics.py` | Strategy identifiers, seed contract, and statistical routines |
| `vn30ppo/__init__.py` | Package imports |
| `vn30ppo/walk_forward.py` | Required import of the original `main.py`; no walk-forward results are reported here |
| `tests/` | Eight selected existing test files for the included implementation and CLI |
| `requirements.txt` | Direct dependency versions |
| `.gitattributes`, `SHA256SUMS` | Preserve and verify frozen research asset bytes |

Scientific source files and stored experimental outputs are retained without
algorithm or numerical changes. Draft manuscripts, document-conversion
scripts, unrelated download utilities, caches, duplicate data downloads, and
unfinished walk-forward outputs are excluded from Git.

## Frozen data

Track only `Data/VN30_VNINDEX_2016_2025.csv` from `Data/`. It is the exact
input snapshot recorded in the experiment configuration, with 68,736 OHLCV
rows, 31 ticker series, and 2,820 distinct trading dates spanning 2014-09-23
through 2025-12-31. Earlier observations support feature lookbacks; the
filename does not restrict raw coverage to 2016 onward.

Columns are `time, open, high, low, close, volume, ticker`. Prices use the
vendor's thousands-of-VND convention; the simulator applies the recorded
`price_multiplier = 1000`. The acquisition source was vnstock with
Vietcap/VCI. No live download or vnstock installation is needed to rerun from
this snapshot. Downloading current vendor data can change the experiment.

The same CSV supplies the static equity universe. It is not a point-in-time
index-membership file. The configuration records ticker order, training
eligibility, and the three equities lacking sufficient training history.

Data SHA-256:

```text
a2fe87a2b1275755e974b558eddf92100342d88518d7f8097b03452e96a9b91f
```

## Stored results supporting the paper

All paths below are relative to `artifacts/`.

| Paper content | Stored source |
| --- | --- |
| Table I: controls and selected parameters | `disagreement_ppo_main/adaptive_rebalance_calibration.json`, `consensus_only_calibration.json`, their two `*_validation_grid.csv` files, and `reviewer_evaluation/matched_control_calibration.csv` |
| Table II: all 17 strategy/benchmark rows | `disagreement_ppo_main/metrics.csv` and `reviewer_evaluation/generic_control_metrics.csv` |
| Table III: differences and 95% intervals | `reviewer_evaluation/paired_metric_bootstrap.csv`; filter `block_length = 20` and metrics `cagr`, `sharpe_ratio`, `total_turnover`, `transaction_cost_vnd` |
| Block-length sensitivity | The same bootstrap CSV, with block lengths 10 and 40 |
| Seed dispersion and leave-one-seed-out diagnostics | `reviewer_evaluation/seed_uncertainty_summary.csv`, `leave_one_seed_out_consensus.csv`, `pairwise_seed_tv_matrix.csv` |
| Disagreement series and exploratory correlations | `reviewer_evaluation/decision_disagreement_diagnostics.csv`, `disagreement_correlations.json` |
| Economic cost saving and Combined-v2 budget activity | `reviewer_evaluation/cost_economic_significance.csv`, `combined_budget_diagnostics.csv` |
| Annualization and newly tradable equities | `reviewer_evaluation/annualization_audit.csv`, `unseen_asset_exposure.csv` |
| Daily NAV, trades, target weights, and realized holdings | The 16 selected `daily_*.csv` files across the two artifact directories |
| Frozen policies and their configuration | `disagreement_ppo_main/ppo_seed_{5,10,15,20,25}.pt`, `experiment_config.json` |
| Stored diagnostic visualizations | The four selected PNGs in `reviewer_evaluation/fig/` |

The bootstrap CSV retains all seven comparisons, three block lengths, and
20,000-replication results. Costs are stored in VND, and returns/drawdowns as
fractions. The stored diagnostic PNGs include the architecture, disagreement
diagnostics, forward-return scatter, and performance/turnover comparison.
Their panel arrangement differs from the cropped figures in the manuscript.

### Strategy names

| Paper label | Identifier in main result files |
| --- | --- |
| PPO mean | `PPO_mean_ensemble` |
| DAR-Soft, the paper's primary execution control | `PPO_soft_disagreement_rebalance` |
| DAR-Combined-v1, development benchmark | `PPO_robust_asset_adaptive_optimizer_frozen_v1` |
| DAR-Combined-v2, calibration-selected optimizer | `PPO_robust_asset_adaptive_optimizer` |
| Consensus-only | `PPO_consensus_only_gate` |

The historical `PROPOSED_STRATEGY` constant and `proposed_strategy` metadata
still identify Combined-v1. They preserve the original pipeline/checkpoint
contract; use the mapping above to identify the paper's primary DAR-Soft
results. Correlation diagnostics combine proposals from separate seed
trajectories and do not isolate the shared-state disagreement used by the gate.

## Environment and inspection

The recorded environment uses Python 3.11.9. The direct dependencies in
`requirements.txt` are pinned to versions read from the research environment.
This is not a full lock of transitive packages, drivers, or system libraries.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows, or
`source .venv/bin/activate` on Linux/macOS, then:

```bash
python -m pip install -r requirements.txt
```

Training was recorded on an NVIDIA GeForce RTX 2050 with PyTorch
`2.13.0+cu126` and CUDA 12.6. The requirements pin the base PyTorch version;
GPU replay also requires a compatible PyTorch/CUDA build. The commands below
use CPU; `--device cuda:0` or `--device auto` selects GPU where available.
Independent training and replay across devices need not be bitwise identical.

Inspecting the CSV/JSON files or PNGs requires no research computation. For
example, these commands display existing metrics without recomputing them:

```bash
python -c "import pandas as pd; print(pd.read_csv('artifacts/disagreement_ppo_main/metrics.csv').to_string(index=False))"
python -c "import pandas as pd; print(pd.read_csv('artifacts/reviewer_evaluation/generic_control_metrics.csv').to_string(index=False))"
```

Verify archived data, checkpoints, and results using the standard library.
Run this Python snippet from the repository root:

```python
from hashlib import sha256
from pathlib import Path

for line in Path('SHA256SUMS').read_text(encoding='utf-8').splitlines():
    expected, filename = line.split('  ', 1)
    assert sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
print('All frozen research files match SHA256SUMS')
```

## Reproduce the research

Run commands from the repository root. These commands perform research
computation. Use new directories under `runs/` to preserve the reported
results under `artifacts/`; `runs/` is excluded from Git.

### Reevaluate the five frozen checkpoints

First copy only the recorded configuration and five checkpoints into a new
run directory. Run this setup snippet in Python from the repository root:

```python
from pathlib import Path
from shutil import copy2

source = Path('artifacts/disagreement_ppo_main')
replay = Path('runs/replay')
replay.mkdir(parents=True, exist_ok=False)
for name in ['experiment_config.json', *[f'ppo_seed_{s}.pt' for s in [5, 10, 15, 20, 25]]]:
    copy2(source / name, replay / name)
```

Then calibrate execution controls on 2023 and reevaluate without PPO
retraining:

```bash
python optimize_ensemble.py --artifacts runs/replay --device cpu
```

### Reproduce control comparisons and statistical analysis

Use the archived checkpoints and daily histories:

```bash
python reviewer_evaluation.py --artifacts artifacts/disagreement_ppo_main --output runs/reviewer_analysis --device cpu --bootstrap-replications 20000
```

This reruns generic-control calibration/backtests and the statistical analyses.
To use the preceding replay instead, replace `--artifacts` with `runs/replay`
and choose a different new output directory. Bootstrap settings include
20,000 resamples and block lengths 10, 20, and 40. The script also creates
diagnostic plots in the output directory's `fig/` subdirectory.

### Retrain from the recorded study settings

The training CLI defaults match the stored study settings. These commands
train all five seeds, select checkpoints using 2022 validation, calibrate
execution controls using 2023, and evaluate on the development period:

```bash
python train_ppo.py --data Data/VN30_VNINDEX_2016_2025.csv --universe Data/VN30_VNINDEX_2016_2025.csv --output runs/retrained --seeds 5,10,15,20,25 --timesteps 200000 --device cpu
python optimize_ensemble.py --artifacts runs/retrained --device cpu
python reviewer_evaluation.py --artifacts runs/retrained --output runs/retrained_analysis --device cpu --bootstrap-replications 20000
```

Existing implementation tests can be run separately:

```bash
python -m pytest tests
```

