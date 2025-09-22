# PPO Training Metrics Guide

This guide explains how to interpret common **Proximal Policy
Optimization (PPO)** reinforcement learning training metrics.

------------------------------------------------------------------------

## 1️⃣ `approx_kl` -- Approximate Kullback--Leibler Divergence

**Definition:** Estimate of how much the new policy diverges from the
old policy after an update.

**Purpose:** PPO constrains updates to avoid abrupt policy changes.

**Interpretation:** - Low (≈0.001--0.01): Healthy, small policy
changes. - Moderate (≈0.01--0.05): Acceptable but monitor. - High
(\>0.05): Too large updates; lower learning rate or clip range.

------------------------------------------------------------------------

## 2️⃣ `entropy_loss` -- Negative Policy Entropy

**Definition:** Negative of the policy's entropy; measures randomness in
action selection.

**Purpose:** Encourages exploration (entropy bonus).

**Interpretation:** - High negative magnitude: High exploration. - Near
zero: Policy converging, low exploration. - Normal trend: Gradual
decline; sudden drop = premature convergence.

------------------------------------------------------------------------

## 3️⃣ `explained_variance`

**Definition:** R²-like measure of how well the value function predicts
returns.

**Purpose:** Evaluates critic accuracy.

**Interpretation:** - Near 1.0: Excellent critic. - 0.0: No better than
constant guess. - Negative: Critic worse than trivial baseline.

------------------------------------------------------------------------

## 4️⃣ `loss` -- Total PPO Loss

**Definition:** Combined objective minimized by PPO:

    loss = policy_loss + c1 * value_loss + c2 * entropy_loss

(where `c1` and `c2` are coefficients).

**Purpose:** Tracks overall optimization.

**Interpretation:** - Should decrease on average. - Large spikes
indicate instability.

------------------------------------------------------------------------

## 5️⃣ `policy_gradient_loss`

**Definition:** Surrogate objective driving policy updates:

    - E[min(r_t * A_t, clip(r_t, 1 - ε, 1 + ε) * A_t)]

where `r_t` is the probability ratio and `A_t` is the advantage.

**Purpose:** Measures how well the policy improves expected return while
respecting PPO clipping.

**Interpretation:** - Starts negative and trends toward zero as learning
stabilizes. - Big oscillations suggest unstable training.

------------------------------------------------------------------------

## 6️⃣ `value_loss`

**Definition:** Mean squared error between predicted and actual returns.

**Purpose:** Ensures accurate state-value predictions for stable
advantage estimates.

**Interpretation:** - Should generally decrease. - Persistent high
values show critic is struggling.

------------------------------------------------------------------------

## Quick Diagnostic Table

  ------------------------------------------------------------------------
  Metric                          Ideal Trend       Red Flag Signs
  ------------------------------- ----------------- ----------------------
  **approx_kl**                   Stable low        Sudden jumps \>0.05 →
                                  (≤0.01--0.03)     reduce LR/clip

  **entropy_loss**                Gradual rise      Sharp collapse early →
                                  toward 0 (less    under-exploration
                                  negative)         

  **explained_variance**          Rises toward 1.0  Stays near 0 or
                                                    negative → critic not
                                                    learning

  **loss (total)**                Gradually         Huge spikes → unstable
                                  decreases         

  **policy_gradient_loss**        Negative,         Wild oscillations →
                                  trending toward 0 unstable

  **value_loss**                  Decreasing        Persistent high values
                                                    → critic struggling
  ------------------------------------------------------------------------

------------------------------------------------------------------------

**Key Takeaway:**\
Monitor **approx_kl** and **entropy_loss** for exploration and policy
stability. Use **explained_variance** and **value_loss** to judge critic
quality. Watch **policy_gradient_loss** and **loss** for overall
optimization health.
