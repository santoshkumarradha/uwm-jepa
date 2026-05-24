# UWM-JEPA: Predictive World Models That Imagine in Belief Space

*Unitary density-matrix latent prediction for partially observable JEPA world models.*

**Santosh Kumar Radha** &nbsp;·&nbsp; **Oktay Goktas** &nbsp;·&nbsp; AgentField AI, Toronto

<p align="center">
  <img src="paper/figures/fig_hero.png" alt="From point latents to belief-structured imagination" width="640"/>
</p>

> Standard vector-latent JEPAs predict a point in representation space. UWM-JEPA represents the latent as a density matrix and rolls it forward on an isospectral orbit. Under partial observability this gives the predictor a structured latent in which uncertainty and hidden modes can be carried through blind rollout; actions steer the unitary trajectory through `H(a) = H₀ + a·H₁`.

## Abstract

World models for partially observed environments must imagine multiple compatible hidden futures and steer between them under counterfactual actions. JEPA-family world models do this in latent space, but a vector-valued latent has no internal structure for carrying the belief over hidden continuations through blind rollout. We introduce UWM-JEPA, a JEPA world model with a density-matrix latent on a joint system–environment space and a learned unitary predictor. The construction preserves the joint-state spectrum exactly during rollout, so the predictor itself cannot dissipate the represented uncertainty.

On a hidden-velocity indicator task requiring five-step forward simulation under a given action sequence with the target observation masked, UWM-JEPA reaches **0.77 accuracy** and degrades monotonically as actions are perturbed; a parameter-matched LSTM-JEPA with the same objective and action head **collapses to majority-class accuracy (0.53)** under every action condition. Under blind rollout, UWM-JEPA loses fewer than ten points of probe *R*² at short horizons while vector-latent baselines lose forty-one and sixty-eight; both nevertheless tie on a held-out context probe, locating the separation in the predictor rather than the encoder. Action sensitivity itself requires training against counterfactual rather than teacher-forced targets, a finding that applies beyond the unitary parameterisation. For JEPA world models to imagine under partial observability, **latent geometry and predictor dynamics matter, not encoder capacity alone**.

## Result

The headline empirical contrast is the **hidden-velocity indicator task**: predict the sign of velocity five steps ahead under a given action sequence, with the corresponding observation masked. The model must roll the latent forward under the action sequence and read the answer from its own imagined state.

| Action condition | UWM-JEPA-CF | LSTM-JEPA-CF |
|---|---|---|
| Correct   | **0.770 ± 0.011** | 0.530 ± 0.000 |
| No-action | 0.733 ± 0.013     | 0.530 ± 0.000 |
| Shuffled  | 0.685 ± 0.003     | 0.530 ± 0.000 |
| Wrong     | 0.639 ± 0.034     | 0.530 ± 0.000 |

*Mean ± std over three seeds. Class distribution is 265 / 235, so majority-class accuracy is 0.53.*

UWM-JEPA-CF degrades monotonically as the action sequence is perturbed. The parameter-matched LSTM-JEPA-CF — same training objective, same action head — predicts the majority class for every one of the 500 held-out examples under every action condition. The two models additionally tie on a held-out context probe (*p* = 0.70), so the separation is in the predictor, not the encoder. Full evidence in Fig. 2 of the paper.

## Repository layout

```
uwm-jepa/
├── paper/                          # Manuscript source and final PDF
│   ├── main.tex                    # Top-level document
│   ├── main.pdf                    # Compiled paper
│   ├── preamble.tex                # Shared LaTeX preamble
│   ├── references.bib              # Bibliography
│   ├── sections/                   # Section sources
│   └── figures/                    # All figures used in the paper
│
└── experiments/                    # Figure regeneration from bundled data
    ├── README.md
    ├── make_figures.py             # Single script regenerates all evidence figures
    └── data/                       # Aggregated outputs from training/eval runs
```

## Reproducing the figures

```bash
cd experiments
python3 make_figures.py
```

This regenerates Figures 2, 3, 4 of the main text and Figures S1, S2 of the supplement from the bundled JSON/CSV under `experiments/data/`. Outputs are written as `.pdf` (used by the paper) and `.png` to `paper/figures/`.

Requires: `numpy`, `matplotlib`. Tested with Python 3.12.

## Building the paper

The paper builds with [Tectonic](https://tectonic-typesetting.github.io/) (single-binary LaTeX engine) or any standard `pdflatex` + `bibtex` toolchain:

```bash
cd paper
tectonic main.tex
```

## Citation

```bibtex
@article{radha2026uwmjepa,
  title  = {UWM-JEPA: Predictive World Models That Imagine in Belief Space},
  author = {Radha, Santosh Kumar and Goktas, Oktay},
  year   = {2026},
  note   = {Preprint. \url{https://github.com/santoshkumarradha/uwm-jepa}}
}
```

## Contact

Santosh Kumar Radha — `contact@santoshkumarradha.com` · `santosh@agentfield.ai`
