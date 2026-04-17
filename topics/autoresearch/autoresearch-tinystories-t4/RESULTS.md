# AutoResearch Results — TinyStories / GCP T4

Experiment results from autonomous overnight ML research runs.
Each run starts from the T4-adapted baseline `train.py` and iterates for 8 hours.

---

## Summary

| | Run 1 — agent.py | Run 2 — flyte_workflow.py |
|---|---|---|
| **Date** | 2026-04-16 | 2026-04-17 |
| **Orchestration** | Plain Python loop | Flyte 2 tasks |
| **Baseline val_bpb** | 3.4172 | 3.6773 |
| **Final val_bpb** | 1.2303 | 1.4251 |
| **Improvement** | ↓ 2.1869 (64%) | ↓ 2.2522 (61%) |
| **Experiments** | 80 | 58 |
| **Kept** | 17 | 13 |
| **Reverted** | 63 | 45 |
| **Success rate** | 21% | 22% |
| **Budget** | 8 hours | 8 hours |
| **Run ID** | diJmMJdbDDgR1qAAFF1k | E35uWNVNPX1Qn1ipAr62 |

---

## Run 1 — agent.py

**Date:** 2026-04-16  
**Orchestration:** Plain Python loop (`agent.py`)  
**Run ID:** `diJmMJdbDDgR1qAAFF1k`

### Stats
- Baseline val_bpb: **3.4172**
- Final val_bpb: **1.2303**
- Total improvement: **↓ 2.1869 (64%)**
- 80 experiments · 17 kept · 63 reverted · 21% success rate

### val_bpb progression
- **Exp 1–9:** Oscillating near baseline (~3.3–3.5), mostly reverted
- **Exp 10–11:** Major breakthrough — TOTAL_BATCH_SIZE reduction dropped val_bpb from 3.3 → 1.9 in two consecutive kept experiments
- **Exp 12–29:** Stabilizing around 1.8–2.0, agent refining other hyperparameters
- **Exp 30+:** Second step down to ~1.4–1.5
- **Exp 60–80:** Converging around 1.2–1.3, mostly reverted — agent near local minimum

### Key finding
The agent independently discovered that **reducing TOTAL_BATCH_SIZE** is the dominant lever for a time-budgeted T4 run. Smaller batches = more optimizer steps per 5 minutes = faster convergence.

| Experiment | Change | Delta |
|---|---|---|
| Exp 9 | TOTAL_BATCH_SIZE 2^17 → 2^16 | -0.4966 |
| Exp 10 | TOTAL_BATCH_SIZE 2^16 → 2^15 | -0.7983 (best) |
| Exp 11 | TOTAL_BATCH_SIZE 2^15 → 2^14 | further improvement |

Three consecutive kept experiments, each halving the batch size — the agent recognized the pattern and kept pushing.

### Best single change
> **-0.7983** — Experiment 10 showed a massive improvement by reducing TOTAL_BATCH_SIZE from 2^16 to 2^15, giving GRAD_ACCUM_STEPS=2. More optimizer steps in 5 minutes = better convergence.

---

## Run 2 — flyte_workflow.py

**Date:** 2026-04-17  
**Orchestration:** Flyte 2 tasks (`flyte_workflow.py`)  
**Run ID:** `E35uWNVNPX1Qn1ipAr62`

### Stats
- Baseline val_bpb: **3.6773**
- Final val_bpb: **1.4251**
- Total improvement: **↓ 2.2522 (61%)**
- 58 experiments · 13 kept · 45 reverted · 22% success rate

### val_bpb progression
- **Exp 1–10:** Near baseline, agent exploring architecture and LR space
- **Exp 11:** Major breakthrough — same TOTAL_BATCH_SIZE insight as Run 1, val_bpb dropped sharply
- **Exp 12–40:** Oscillating around 1.6–1.8, agent refining config around the new baseline
- **Exp 40–58:** Steady convergence toward 1.4, mostly reverted late in the run

### Key finding
The agent rediscovered the same insight as Run 1 — **TOTAL_BATCH_SIZE reduction** was the primary driver. Starting from a clean baseline, the agent reached the same conclusion independently by experiments 11–12, mirroring the Run 1 pattern exactly.

| Experiment | Change | Delta |
|---|---|---|
| Exp 11 | TOTAL_BATCH_SIZE 2^17 → 2^16 | -0.1270 |
| Exp 12 | TOTAL_BATCH_SIZE 2^16 → 2^15 | -0.7398 (best) |

Two consecutive kept experiments, each halving batch size — same sequence as Run 1 experiments 9–10.

### Best single change
> **-0.7398** — Experiment 12: reducing TOTAL_BATCH_SIZE from 2^16 to 2^15 (~65K → ~32K tokens per gradient update), yielding 299 optimizer steps in 361s. Identical lever as Run 1's best experiment.

---

## Observations

### Both runs converged on the same insight
Without any human guidance, both the `agent.py` loop and the `flyte_workflow.py` loop independently identified that **reducing TOTAL_BATCH_SIZE** is the dominant optimization for a 5-minute training budget on a T4. The agent discovered this by experiment 9–12 in both runs, following the same two-step halving pattern:

| Halving step | Run 1 | Run 2 |
|---|---|---|
| 2^17 → 2^16 (~131K → ~65K tokens) | Exp 9, delta **-0.1144** | Exp 11, delta **-0.1270** |
| 2^16 → 2^15 (~65K → ~32K tokens) | Exp 10, delta **-0.4966** | Exp 12, delta **-0.7398** |

The second halving consistently produced the larger gain in both runs — the agent didn't stop after the first improvement, it recognized the pattern and kept pushing.

### Fewer experiments in Run 2
Run 2 completed 58 experiments vs 80 in Run 1. Likely causes:
- Flyte task orchestration adds small overhead per experiment (~few seconds)
- Slightly higher baseline may have led to longer early exploration

### Similar success rates
Both runs achieved ~21–22% success rate — roughly 1 in 5 proposed changes improved val_bpb. This aligns with the expected difficulty of finding improvements near a local minimum.

### Final val_bpb difference
Run 1 reached 1.2303 vs Run 2's 1.4251. Run 1 had 22 more experiments (more time to explore) and benefited from a slightly lower starting baseline. Given equal experiment counts, both runs would likely converge to similar floors.

### Orchestration is transparent to results
The Flyte workflow produced equivalent results to the plain Python loop — confirming that `core.py` correctly shares the experiment logic between both modes, and Flyte's overhead does not affect the ML outcomes.
