# Inference Equivalence (IE)

This repository contains simulation code for evaluating **Inference Equivalence (IE)**, a framework for representation sufficiency defined through invariance of achiev-
able inference risk within a constrained decoder family.

---

## Core Idea

We study whether preserving low-order statistical structure (e.g., means, covariance) is sufficient to preserve **computational inference ability**.

Instead of comparing representations descriptively, IE evaluates:

> **Minimum achievable inference risk under a specified decoder family.**

This provides a direct test of whether the preserved statistics are insufficient to sustain task-relevant inference.

---

## Key Quantities

We compute two complementary diagnostics:

### 1. Inference Equivalence (IE)

$$
\Delta_{\mathrm{IE}} = R_Y^{(\mathcal{F},* )} - R_X^{(\mathcal{F},* )}
$$

where:

$R^{\mathcal{F},*}$ is the minimum achievable risk under decoder family $\mathcal{F}$  
$X$: original representation  
$Y$: transformed representation

---

### 2. Transfer Loss

Evaluates stability of a fixed decoder trained on \(X\) and tested on transformed data \(Y\), measuring sensitivity of learned decision rules under distribution shift.

---

## Decoder Families

All experiments support multiple decoder classes:

- **linear**: L2-regularized logistic regression
- **quadratic**: QDA with regularization parameter `reg_param = 0.5`
- **both**: full comparison across decoder families

---

## Representation Transformations

We evaluate controlled perturbations on representations:

- Mean-only surrogate (removes variability)
- IID marginal resampling (breaks dependencies)
- Covariance-matched Gaussian model
- Latent factor model (low-rank structure)
- Gaussian copula surrogate (nonlinear dependence preservation)
- Decision-aligned perturbations (geometry-aligned interventions)

Each transformation selectively modifies statistical structure while preserving task labels.

---
## Installation

Clone the repository and install the required Python dependencies before running the simulations:

```bash
git clone https://github.com/anon-research-sketch/Inference-Equivalence.git
cd Inference-Equivalence
# General (macOS, Linux, or Windows with PATH configured)
python -m pip install -r requirements.txt

# Windows users (if 'python' command is not recognized)
py -m pip install -r requirements.txt
```

If the repository is downloaded as a ZIP archive from GitHub, the extracted folder name may appear as `Inference-Equivalence-main`.

```
## Repository Structure

- `run_simulation.py`: main simulation runner
- `figure_IE.py`: generates Fig. 3 / Fig. S1 / Fig. S3–S10
- `figure_transfer.py`: generates Fig. 4 / Fig. S2 / Fig. S11–S18
- `ie_interventions.py`: Controlled perturbations and descriptive benchmark metrics
- `ie_evaluation.py`: IE and transfer evaluation pipeline
- `requirements.txt`: Python dependencies

Simulation outputs are saved under the `ie_sim/` directory. Each run is executed across multiple random seeds, with results saved incrementally after each seed. The final `.pkl` files aggregate results across seeds and are used for figure generation.

If the `python` command is not available on Windows systems, replace it with `py`.

---
## How to Run

Supported task types:

- `linear` (**main-text task**)
- `nonlinear`
- `noise_interaction`
- `rotated`
- `sparse`

Supported decoder options:

- `linear`
- `quadratic`
- `both`

```bash
# Full analysis (all tasks, both decoders)
python run_simulation.py --type all --decoder both
# Fig. 3
python figure_IE.py --results "ie_sim/ie_sim_linear/ie_results_linear_quadratic.pkl" --decoder linear
# Fig. 4
python figure_transfer.py --results "ie_sim/ie_sim_linear/ie_results_linear_quadratic.pkl" --decoder linear
# Fig. S1
python figure_IE.py --results "ie_sim/ie_sim_linear/ie_results_linear_quadratic.pkl" --decoder quadratic
# Fig. S2
python figure_transfer.py --results "ie_sim/ie_sim_linear/ie_results_linear_quadratic.pkl" --decoder quadratic


# Example: single task, single decoder
# task nonlinear, decoder quadratic
python run_simulation.py --type nonlinear --decoder quadratic
python figure_IE.py --results "ie_sim/ie_sim_nonlinear/ie_results_quadratic.pkl" --decoder quadratic
python figure_transfer.py --results "ie_sim/ie_sim_nonlinear/ie_results_quadratic.pkl" --decoder quadratic



```
## Reproducibility

All simulations are fully reproducible from the provided scripts and saved result files.  
Running the commands above will regenerate the main-text and supplementary figures.


