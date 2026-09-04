"""Generates the Model H vs. Model L intrinsic-metric comparison chart
(Phase 2 report). Reads each language's already-computed
statistics/phase2_eval_outputs/results_summary.json and calls the
existing plot_ppl_bpb_comparison() (hindi/eval/plotting.py) -- unlike
every other Phase 2 plot, this one is inherently cross-language, so it
cannot be produced inside either language's single-language Kaggle
training run and is generated here instead, locally, once both
languages' results exist.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "hindi", "eval"))

from plotting import plot_ppl_bpb_comparison  # noqa: E402

with open(os.path.join(ROOT, "hindi", "statistics", "phase2_eval_outputs", "results_summary.json")) as f:
    hindi_results = json.load(f)
with open(os.path.join(ROOT, "nepali", "statistics", "phase2_eval_outputs", "results_summary.json")) as f:
    nepali_results = json.load(f)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppl_bpb_comparison.png")
plot_ppl_bpb_comparison(hindi_results["ppl_bpb"], nepali_results["ppl_bpb"], out_path)
print(f"wrote {out_path}")
print(f"hindi:  perplexity={hindi_results['ppl_bpb']['perplexity']:.3f}  bpb={hindi_results['ppl_bpb']['bits_per_byte']:.4f}")
print(f"nepali: perplexity={nepali_results['ppl_bpb']['perplexity']:.3f}  bpb={nepali_results['ppl_bpb']['bits_per_byte']:.4f}")
