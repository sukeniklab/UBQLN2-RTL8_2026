import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d as _interp1d

plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.sans-serif'] = 'Arial'

# ── Config ────────────────────────────────────────────────────────────────────
# Each construct now carries an explicit `theo_color` for its theoretical ensemble.

CONSTRUCTS = [
    dict(label="UBQLN2_RTL8",             exp="A_S_CC13_015.dat",
         color="grey", theo_color="orange", theo_zorder=10, display="UBQLN2 - RTL8 - STI1 bound"),
    dict(label="UBQLN2_STI1-RTL8_bound",  exp="A_S_CC13_015.dat",
         color="grey", theo_color="red", theo_zorder=9, display="STI1-1 - RTL8 bound"),
    dict(label="UBQLN2_UBL-RTL8_bound",   exp="A_S_CC13_015.dat",
         color="grey", theo_color="blue",theo_zorder=9, display="UBL - RTL8 bound"),
]

TRIALS       = ["trial1", "trial2", "trial3", "trial4", "trial5"]
FITS_BASE    = "foxs_profiles_new"
RECON_SUFFIX = "_avg.dat"   # per-trial average theoretical profile 
N_SCATTER    = 50

SCALE_TO_EXP = True

QMIN, QMAX   = 0.0, 0.30
FS_LABEL     = 14
FS_LEGEND    = 9
LOWQ_IDX_START = 40
LOWQ_IDX_END   = 50



def normalize_to_exp_lowq(I_sim, q_sim, I_exp, q_exp,
                          idx_start=LOWQ_IDX_START, idx_end=LOWQ_IDX_END):
    ### Scale the simulated profile to match the experimental profile in the low-q region.
    fi     = _interp1d(q_sim, I_sim, kind="cubic",
                       bounds_error=False, fill_value="extrapolate")
    I_at_q = fi(q_exp)
    return np.mean(I_exp[idx_start:idx_end]) / np.mean(I_at_q[idx_start:idx_end])


def load_exp(path, qmin=QMIN, qmax=QMAX):
    delim = "," if path.endswith(".csv") else None
    skip = 0
    with open(path) as fh:
        for i, line in enumerate(fh):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                float(s.split(delim)[0])
                skip = i
            except ValueError:
                skip = i + 1
            break
    data = np.loadtxt(path, comments="#", delimiter=delim, skiprows=skip)
    q, I, sigma = data[:, 0], data[:, 1], data[:, 2]
    mask = (q >= qmin) & (q <= qmax) & (sigma > 0) & (I > 0)
    q, I, sigma = q[mask], I[mask], sigma[mask]
    return q, I / I[0], sigma / I[0]


def load_recon_trials(label, trials=TRIALS, fits_base=FITS_BASE):
    ## Load FOXS output
    all_I         = []
    q_ref         = None
    loaded_trials = []

    for trial in trials:
        recon_path = f"{fits_base}/{label}/{trial}/{label}_{trial}{RECON_SUFFIX}"
        print(f"  Loading {trial}/{label} from {recon_path}")
        try:
            d = np.loadtxt(recon_path, comments="#")
        except FileNotFoundError:
            print(f"  WARNING: not found, skipping — {recon_path}")
            continue

        q_src, I_src = d[:, 0], d[:, 1]

        if q_ref is None:
            q_ref = q_src
        elif not np.allclose(q_src, q_ref, rtol=1e-4):
            print(f"  WARNING: q-grid mismatch for {trial}/{label} — skipping")
            continue

        all_I.append(I_src)
        loaded_trials.append(trial)

    if not all_I:
        raise RuntimeError(f"No trial data found for {label}")

    all_I = np.array(all_I)
    return q_ref, all_I, all_I.mean(axis=0), all_I.std(axis=0, ddof=1), loaded_trials



def compute_construct(c):
    ### Load, scale, and derive residuals for one construct.
    
    q_exp, I_exp, sigma_exp = load_exp(c["exp"])
    q_rec, all_I_trials, I_rec_mean, I_rec_std, loaded_trials = load_recon_trials(c["label"])
    scale = normalize_to_exp_lowq(I_rec_mean, q_rec, I_exp, q_exp) if SCALE_TO_EXP else 1.0
   
    I_rec_mean = scale * I_rec_mean
    I_rec_std  = scale * I_rec_std

    fi    = _interp1d(q_rec, I_rec_mean, kind="cubic",
                      bounds_error=False, fill_value="extrapolate")
    resid = (I_exp - fi(q_exp)) / sigma_exp

    dof      = max(1, len(q_exp) - (1 if SCALE_TO_EXP else 0))
    chi2_red = np.sum(resid**2) / dof

    return dict(
        c=c,
        q_exp=q_exp, I_exp=I_exp, sigma_exp=sigma_exp,
        q_rec=q_rec, I_rec_mean=I_rec_mean, I_rec_std=I_rec_std,
        resid=resid, chi2_red=chi2_red,
    )


results = [compute_construct(c) for c in CONSTRUCTS]
n = len(results)

### GRID FIGURE WITH RESIDUALS BELOW EACH PANEL ###
fig, axes = plt.subplots(
    2, n,
    figsize=(4.5 * n, 4.5),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0, "wspace": 0.12},
    squeeze=False,           
)

for i, r in enumerate(results):
    ax_top = axes[0, i]
    ax_bot = axes[1, i]

    c          = r["c"]
    color      = c["color"]
    theo_color = c["theo_color"]
    display    = c["display"]

    q_exp, I_exp, sigma_exp     = r["q_exp"], r["I_exp"], r["sigma_exp"]
    q_rec, I_rec_mean, I_rec_std = r["q_rec"], r["I_rec_mean"], r["I_rec_std"]

    eb_step = max(1, len(q_rec) // N_SCATTER)
    eb_idx  = np.arange(0, len(q_rec), eb_step)
    yerr_lo = np.minimum(I_rec_std[eb_idx], I_rec_mean[eb_idx] * 0.999)

    # Experimental data
    ax_top.errorbar(q_exp, I_exp, yerr=sigma_exp,
                    fmt="o", ms=2, color=color,
                    elinewidth=0.5, capsize=0, alpha=0.35,
                    label="Experimental", zorder=4)

    # Theoretical mean ± std, now colored per construct
    ax_top.errorbar(q_rec[eb_idx], I_rec_mean[eb_idx],
                    yerr=[yerr_lo, I_rec_std[eb_idx]],
                    fmt="s", ms=4,
                    markerfacecolor="none", markeredgecolor=theo_color, markeredgewidth=0.9,
                    elinewidth=1.0, capsize=2, alpha=0.9,
                    ecolor=theo_color, label="Theoretical Ensemble", zorder=5)

    ax_top.set_yscale("log")
    ax_top.set_ylim(1e-4, 2)
    ax_top.set_xlim(QMIN, QMAX)
    ax_top.tick_params(axis="both", labelsize=FS_LABEL)
    ax_top.set_xticklabels([])
    ax_top.legend(fontsize=FS_LEGEND, loc="lower left", frameon=False)
    ax_top.text(0.97, 0.97,
                f"{display}",
                transform=ax_top.transAxes,
                ha="right", va="top", fontsize=FS_LEGEND + 1, color=theo_color)

    # Residuals panel
    ax_bot.axhline(0, color="k", linestyle="--", lw=1.5, zorder=1)
    ax_bot.scatter(q_exp, r["resid"], s=4, color=color, alpha=0.4, zorder=3)
    ax_bot.set_ylim(-35, 35)
    ax_bot.set_xlim(QMIN, QMAX)
    ax_bot.set_yticks([-25, 0,25])
    ax_bot.tick_params(axis="both", labelsize=FS_LABEL - 2)
    ax_bot.set_xlabel(r"$q$ (Å$^{-1}$)", fontsize=FS_LABEL)

    if i == 0:
        ax_top.set_ylabel(r"$I(q)$", fontsize=FS_LABEL, labelpad=4)
        ax_bot.set_ylabel(r"$\Delta/\sigma$", fontsize=FS_LABEL - 2, labelpad=4)
    else:
        ax_top.set_yticklabels([])
        ax_bot.set_yticklabels([])

fig.subplots_adjust(hspace=0, wspace=0.12)
fig.savefig("Fig4F.svg", dpi=300, bbox_inches="tight")
fig.savefig("Fig4F.png", dpi=150, bbox_inches="tight")
plt.close(fig)


### ALL CONSTRUCTS OVERLAID IN ONE PANEL ###
fig2, (ax_top, ax_bot) = plt.subplots(
    2, 1,
    figsize=(5.0, 5.0),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
)
 
# Only plot exp data once if all constructs share the same experimental file
exp_paths  = [c["exp"] for c in CONSTRUCTS]
single_exp = len(set(exp_paths)) == 1
 
if single_exp:
    r0 = results[0]
    ax_top.errorbar(r0["q_exp"], r0["I_exp"], yerr=r0["sigma_exp"],
                    fmt="o", ms=2, color="grey",
                    elinewidth=0.5, capsize=0, alpha=0.30,
                    label="Experimental", zorder=2)
 
for r in results:
    c          = r["c"]
    theo_color = c["theo_color"]
    display    = c["display"]
    order=c["theo_zorder"]
 
    q_rec, I_rec_mean, I_rec_std = r["q_rec"], r["I_rec_mean"], r["I_rec_std"]
 
    if not single_exp:
        ax_top.errorbar(r["q_exp"], r["I_exp"], yerr=r["sigma_exp"],
                        fmt="o", ms=2, color=c["color"],
                        elinewidth=0.5, capsize=0, alpha=0.25, zorder=2)
 
    # Theoretical mean ± std, same open-square error-bar style as the grid figure
    eb_step = max(1, len(q_rec) // N_SCATTER)
    eb_idx  = np.arange(0, len(q_rec), eb_step)
    yerr_lo = np.minimum(I_rec_std[eb_idx], I_rec_mean[eb_idx] * 0.999)
 
    ax_top.errorbar(q_rec[eb_idx], I_rec_mean[eb_idx],
                    yerr=[yerr_lo, I_rec_std[eb_idx]],
                    fmt="s", ms=4,
                    markerfacecolor="none", markeredgecolor=theo_color, markeredgewidth=0.9,
                    elinewidth=1.0, capsize=2, alpha=0.9,
                    ecolor=theo_color, zorder=order,
                    label=f"{display}")
 
    ax_bot.scatter(r["q_exp"], r["resid"], s=4, color=theo_color, alpha=0.5, zorder=order)
 
ax_top.set_yscale("log")
ax_top.set_ylim(1e-4, 2)
ax_top.set_xlim(QMIN, QMAX)
ax_top.set_ylabel(r"$I(q)$", fontsize=FS_LABEL, labelpad=4)
ax_top.tick_params(axis="both", labelsize=FS_LABEL)
ax_top.set_xticklabels([])
ax_top.legend(fontsize=FS_LEGEND, loc="lower left", frameon=False)
 
ax_bot.axhline(0, color="k", linestyle="--", lw=1.5, zorder=1)
ax_bot.set_ylim(-35, 35)
ax_bot.set_xlim(QMIN, QMAX)
ax_bot.set_yticks([-25, 0,25])
ax_bot.tick_params(axis="both", labelsize=FS_LABEL - 2)
ax_bot.set_ylabel(r"$\Delta/\sigma$", fontsize=FS_LABEL - 2, labelpad=4)
ax_bot.set_xlabel(r"$q$ (Å$^{-1}$)", fontsize=FS_LABEL)
 
fig2.subplots_adjust(hspace=0)
fig2.savefig("Fig4F_overlaid.svg", dpi=300, bbox_inches="tight")
fig2.savefig("Fig4F_overlaid.png", dpi=150, bbox_inches="tight")
plt.close(fig2)