# Hidden Markov Model with Macro‑Driven Transition Probabilities

Implements a Hidden Markov Model where the transition probabilities between hidden regimes depend on macro variables (VIX, DXY, yields). The per‑ETF score is the filtered probability of being in the highest‑mean regime at the last time step – a regime‑aware alpha signal.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Number of hidden states configurable (default 3)
- Emission: Gaussian with state‑specific mean and variance
- Transition probabilities: multinomial logistic regression on macro variables
- Score = probability of best regime given macro history
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-hmm-macro-transitions-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast, EM converges quickly)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High score → ETF is likely in a favourable regime (high expected return) given current macro conditions.

## Requirements

See `requirements.txt`.
