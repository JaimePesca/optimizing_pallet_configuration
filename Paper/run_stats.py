import sys, numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

historical = np.array([0.812, 0.811, 0.862, 0.838, 0.833, 0.848, 0.833, 0.825,
                       0.854, 0.829, 0.841, 0.852, 0.886, 0.831, 0.834, 0.854])
bl   = np.array([0.886, 0.895, 0.906, 0.889, 0.900, 0.895, 0.901, 0.895,
                 0.883, 0.866, 0.883, 0.833, 0.908, 0.889, 0.900, 0.890])
baf  = np.array([0.845, 0.832, 0.808, 0.800, 0.837, 0.831, 0.835, 0.838,
                 0.834, 0.836, 0.830, 0.829, 0.803, 0.831, 0.832, 0.840])
blsf = np.array([0.791, 0.780, 0.820, 0.786, 0.790, 0.800, 0.806, 0.795,
                 0.799, 0.828, 0.830, 0.828, 0.690, 0.818, 0.810, 0.834])

print("=== DESCRIPTIVE STATISTICS (n=16 flights) ===")
for name, data in [("Historical", historical), ("BL", bl), ("BAF", baf), ("BLSF", blsf)]:
    print(f"{name:12s}  mean={data.mean()*100:.2f}%  std={data.std(ddof=1)*100:.2f}  "
          f"min={data.min()*100:.2f}  max={data.max()*100:.2f}  median={np.median(data)*100:.2f}")

print()
print("=== WILCOXON SIGNED-RANK TESTS (one-sided: heuristic > Historical) ===")
wilcoxon_results = []
for name, data in [("BL", bl), ("BAF", baf), ("BLSF", blsf)]:
    diff = data - historical
    stat, p = stats.wilcoxon(diff, alternative='greater')
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    wilcoxon_results.append({"name": name, "W": stat, "p": p, "sig": sig, "delta": diff.mean()*100})
    print(f"{name} vs Historical: W={stat:.1f}, p={p:.6f} {sig}, mean diff={diff.mean()*100:+.2f} pp")

print()
print("=== LATEX TABLE ===")
print()

lines = []
lines.append(r"\begin{table}[ht]")
lines.append(r"\caption{Comparative volumetric utilisation across 16 charter flights (values in \%).")
lines.append(r"Statistical significance of improvement over the historical baseline")
lines.append(r"assessed via one-sided Wilcoxon signed-rank test ($n=16$, $\alpha=0.05$).}")
lines.append(r"\label{tab:comparative}")
lines.append(r"\centering")
lines.append(r"\begin{tabular}{lrrrrrrr}")
lines.append(r"\toprule")
lines.append(r"Method & Mean & Std & Min & Max & Median & $\Delta$\,mean & $p$-value \\")
lines.append(r"       &      &     &     &     &        & (pp) & \\")
lines.append(r"\midrule")

best_mean = max(bl.mean(), baf.mean(), blsf.mean()) * 100

for name, data, wrow in [
    ("Historical (baseline)", historical, None),
    ("Bottom-Left (BL)",      bl,   wilcoxon_results[0]),
    ("Best Area Fit (BAF)",   baf,  wilcoxon_results[1]),
    ("Best Long Side Fit (BLSF)", blsf, wilcoxon_results[2]),
]:
    is_best = abs(data.mean()*100 - best_mean) < 0.01
    mean_s = (r"\textbf{" + f"{data.mean()*100:.2f}" + "}") if is_best else f"{data.mean()*100:.2f}"
    std_s  = f"{data.std(ddof=1)*100:.2f}"
    min_s  = f"{data.min()*100:.2f}"
    max_s  = f"{data.max()*100:.2f}"
    med_s  = f"{np.median(data)*100:.2f}"
    if wrow is None:
        delta_s, p_s = "---", "---"
    else:
        delta_s = f"{wrow['delta']:+.2f}"
        p_val, sig = wrow['p'], wrow['sig']
        p_s = f"{p_val:.4f}\\,{sig}" if p_val >= 0.0001 else f"$<$0.0001\\,{sig}"
    lines.append(f"{name} & {mean_s} & {std_s} & {min_s} & {max_s} & {med_s} & {delta_s} & {p_s} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\begin{tablenotes}")
lines.append(r"  \small")
lines.append(r"  \item pp = percentage points vs.\ historical mean.")
lines.append(r"  \item Significance: *** $p<0.001$, ** $p<0.01$, * $p<0.05$, ns = not significant.")
lines.append(r"\end{tablenotes}")
lines.append(r"\end{table}")

for l in lines:
    print(l)
