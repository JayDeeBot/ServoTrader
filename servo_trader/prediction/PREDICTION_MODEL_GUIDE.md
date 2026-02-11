# ServoTrader Prediction Model v2.5 - On-Chain Data Integration

## Overview

Version 2.5 adds **free on-chain blockchain metrics** from Blockchain.com's Charts API. Research shows that on-chain data is critical for achieving high daily prediction accuracy (up to 82% vs ~55% with technical indicators alone).

## New Features

### On-Chain Metrics (from Blockchain.com - FREE, no API key required)

| Metric | Description | Predictive Value |
|--------|-------------|-----------------|
| `hash-rate` | Network hash rate (TH/s) | Network security/miner confidence |
| `n-unique-addresses` | Active addresses per day | Adoption/activity level |
| `n-transactions` | Confirmed transactions | Network usage |
| `estimated-transaction-volume-usd` | USD value transferred | Economic activity |
| `difficulty` | Mining difficulty | Network health |
| `transaction-fees-usd` | Total fees in USD | Network demand |
| `miners-revenue` | Total miner revenue | Economic health |

### Derived On-Chain Features

For each base metric, the system computes:
- **1-day and 7-day percentage changes** - Momentum signals
- **7-day and 30-day MA ratios** - Trend signals
- **7-day momentum** - Direction of change
- **7-day volatility** - Stability measure

Plus cross-metric ratios:
- `tx_per_address` - Average transactions per active user
- `avg_tx_value` - Average transaction size
- `avg_fee_per_tx` - Fee pressure indicator
- `revenue_per_hash` - Mining profitability
- `hashrate_growth_7d` - Network growth rate

**Total new features: ~50 on-chain features**

## Installation

```bash
# Install required packages (if not already installed)
pip install requests pandas numpy scikit-learn lightgbm boruta optuna pyyaml
```

## Usage

### Quick Test (no optimization)

```bash
# Daily prediction with all features
python train_prediction_model.py --horizon 1440 --no-boruta --no-optuna

# Without on-chain data (for comparison)
python train_prediction_model.py --horizon 1440 --no-boruta --no-optuna --no-onchain
```

### Full Training (with optimization)

```bash
# Full optimization with Boruta + Optuna
python train_prediction_model.py --horizon 1440

# More Optuna trials for better hyperparameters
python train_prediction_model.py --horizon 1440 --optuna-trials 100
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--horizon` | 60 | Prediction horizon in minutes (1440 = 1 day) |
| `--threshold` | 0.001 | Direction threshold (0.1%) |
| `--no-sentiment` | False | Disable Fear & Greed sentiment |
| `--no-onchain` | False | Disable on-chain blockchain features |
| `--no-boruta` | False | Disable Boruta feature selection |
| `--no-optuna` | False | Disable Optuna optimization |
| `--boruta-trials` | 100 | Max Boruta iterations |
| `--optuna-trials` | 50 | Number of Optuna trials |

## File Structure

```
servo_trader/prediction/
├── __init__.py              # Module exports
├── onchain_data.py          # On-chain data fetcher (NEW)
└── train_prediction_model.py # Training script (v2.5)
```

## Data Caching

On-chain data is automatically cached to `~/.servo_trader/onchain_cache/` to:
- Reduce API calls
- Speed up repeated training runs
- Allow offline training after initial fetch

Cache is refreshed after 24 hours by default.

## Expected Results

Based on research and the nature of on-chain data:

| Configuration | Expected Accuracy |
|---------------|-------------------|
| Technical indicators only | 50-55% |
| + Fear & Greed sentiment | 54-56% |
| + On-chain data | **60-70%** |
| + CNN-LSTM architecture | 75-82% |

The on-chain features capture **daily-scale market dynamics** that technical indicators miss:
- Institutional activity (large transaction volumes)
- Network adoption trends (active addresses)
- Miner behavior (hash rate, revenue)
- Market demand (transaction fees)

## Troubleshooting

### "On-chain data module not found"

Ensure both files are in the same directory:
```
servo_trader/prediction/
├── onchain_data.py
└── train_prediction_model.py
```

Or copy `onchain_data.py` to the same directory as the training script.

### "Network request failed"

The Blockchain.com API is free but may occasionally be slow. The script includes:
- 30-second timeout
- Automatic retry from cache on failure
- Rate limiting (0.5s between requests)

### Missing features in merged data

If on-chain data doesn't cover your full date range, features are forward/backward filled. This is expected for dates before Bitcoin's early history.

## Version History

- **v2.5**: Added on-chain data integration (Blockchain.com API)
- **v2.4**: Daily resampling support, improved technical indicators
- **v2.3**: Optuna hyperparameter optimization
- **v2.2**: Boruta feature selection
- **v2.1**: Fear & Greed sentiment integration
- **v2.0**: LightGBM base model

## Next Steps (Phase 3)

If on-chain features achieve 60-70% accuracy, consider:
1. **CNN-LSTM architecture** - Capture temporal patterns in on-chain data
2. **Additional on-chain metrics** - MVRV, NVT, SOPR from paid APIs
3. **Social sentiment** - Twitter/Reddit sentiment analysis