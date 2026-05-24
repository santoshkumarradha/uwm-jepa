# Experiments

This folder regenerates the evidence figures in the paper from the experiment outputs bundled under `data/`.

## Regenerate the figures

```bash
python3 make_figures.py
```

Figures are written to `../paper/figures/` as both `.pdf` (used by the paper) and `.png`.

Requires: `numpy`, `matplotlib`. Tested with Python 3.12.

## Figure → data mapping

| Figure (paper) | Data file(s) used | Script function |
|---|---|---|
| Fig. 2 — Action-binding evidence | `multiseed_cf.json`, `actionctrl_multiseed.json`, `indicator.json`, `indicator_lstm_cf_diagnostics.json` | `make_action_binding` |
| Fig. 3 — Blind-rollout evidence | `fairhorizon_retrieval.json`, `residual_predictor_retrieval.json`, `retention_3panel.json` | `make_blind_rollout` |
| Fig. 4 — Held-out probe | `multiseed_heldout_probe.json` | `make_heldout_probe` |
| Fig. S1 — Invariants verification | `invariants.csv` | `make_invariants` |
| Fig. S2 — Collapse phase | `multiseed_dashboard.json` | `make_collapse_phase` |

Fig. 1a (concept) and Fig. 1b (architecture) are handcrafted and shipped as static PNG in `../paper/figures/`.

## Data files

All files in `data/` are the aggregated outputs of training and evaluation runs. Each is small (a few KB to a few MB) so the entire reproducibility bundle fits in the repo.
