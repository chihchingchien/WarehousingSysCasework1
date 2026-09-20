"""
Task 2 - Demand with seasonality and uncertainty
Cells reported here: Y-A-# / Y-A-$  and  D-D-#  (daily demand by 3-digit ZIP)

Method (one 'experiment' = one scenario k = 1..N_SCEN):
  1. Draw market growth g_m and Tsukumo share growth g_s by inverse-transform
     sampling from triangular distributions.
  2. Perturb week-of-year (CV 20%), day-of-week (CV 15%) and ZIP3 PMF (CV 15%),
     then renormalise each so it still sums to 1.
  3. d(k, week, dow, zip3) = D_k * week_k[w] * dow_k[j] * pmf_k[z]
  4. Aggregate over the 25 scenarios -> mean, mode, sigma, X%-min / X%-max
     and the r-robust bound  d_bar = d + z_r * sigma.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ----------------------------------------------------------------------
# 0. Parameters
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "plot")
os.makedirs(PLOT_DIR, exist_ok=True)

N_SCEN = 25
SEED = 2026
USE_STRATIFIED = True      # stratified u (still inverse-transform); False = plain uniform u

BASE_MARKET = 2_000_000
MKT_G = (0.04, 0.075, 0.12)        # triangular (min, mode, max) market growth
BASE_SHARE = 0.036
SHARE_G = (0.15, 0.20, 0.25)       # triangular (min, mode, max) share growth
PRICE = 3_000

CV_WEEK, CV_DOW, CV_PMF = 0.20, 0.15, 0.15
Z_LEVELS = {"68% (1σ)": 1.0, "95% (1.65σ)": 1.65, "99% (2.33σ)": 2.33}
TOP_ZIP = 10                       # ZIP3s reported in the D-D table

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
def read_table(path):
    """Read csv/tsv (delimiter detected from the header line), strip BOM."""
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline()
    sep = "\t" if "\t" in first else ","
    df_ = pd.read_csv(path, dtype=str, sep=sep, encoding="utf-8-sig")
    df_.columns = [c.replace("\ufeff", "").strip() for c in df_.columns]
    return df_


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip(),
                         errors="coerce")


def load_seasonality(path):
    raw = read_table(path)
    cols = list(raw.columns)
    wi = next(i for i, c in enumerate(cols) if c.lower().startswith("week"))
    di = next(i for i, c in enumerate(cols) if c.lower().startswith("day"))
    wk = pd.DataFrame({"idx": to_num(raw.iloc[:, wi]),
                       "p": to_num(raw.iloc[:, wi + 1])}).dropna().sort_values("idx")
    dw = pd.DataFrame({"idx": to_num(raw.iloc[:, di]),
                       "p": to_num(raw.iloc[:, di + 1])}).dropna().sort_values("idx")
    print(f"Weeks: {len(wk)} (proportion sum = {wk['p'].sum():.4f}) | "
          f"Days: {len(dw)} (proportion sum = {dw['p'].sum():.4f})")
    return (wk["p"].values / wk["p"].sum()), (dw["p"].values / dw["p"].sum())


base_wk, base_dow = load_seasonality(os.path.join(BASE_DIR, "demand_seasonalities.csv"))
W, D7 = len(base_wk), len(base_dow)
T = W * D7                                       # modelled days per year (52*7 = 364)

zm = read_table(os.path.join(BASE_DIR, "zip3_market.csv"))
zp = read_table(os.path.join(BASE_DIR, "zip3_pmf.csv"))
zm["zip3"] = zm["ZIP3"].str.strip().str.zfill(3)
zp["zip3"] = zp["ZIP3"].str.strip().str.zfill(3)
zp["pmf"] = pd.to_numeric(zp["PMF"])
zinfo = (zm[["zip3", "Market", "State"]].drop_duplicates("zip3")
         .merge(zp[["zip3", "pmf"]], on="zip3", how="left"))
zinfo["pmf"] = zinfo["pmf"].fillna(0.0)
zinfo = zinfo[zinfo["pmf"] > 0].reset_index(drop=True)        # ZIP3s with demand
zip_ids = zinfo["zip3"].values
base_pmf = zinfo["pmf"].values / zinfo["pmf"].sum()
print(f"ZIP3 with positive demand: {len(zinfo)} | modelled days: {T}")

# ----------------------------------------------------------------------
# 2. Sampling tools
# ----------------------------------------------------------------------
def tri_inv(u, a, c, b):
    """Inverse CDF of Triangular(min=a, mode=c, max=b)."""
    fc = (c - a) / (b - a)
    return np.where(u < fc,
                    a + np.sqrt(u * (b - a) * (c - a)),
                    b - np.sqrt((1 - u) * (b - a) * (b - c)))


def tri_pdf(x, a, c, b):
    x = np.asarray(x, float)
    return np.where((x >= a) & (x < c), 2 * (x - a) / ((b - a) * (c - a)),
           np.where((x >= c) & (x <= b), 2 * (b - x) / ((b - a) * (b - c)), 0.0))


def draw_u(rng, n):
    if USE_STRATIFIED:                            # one draw per equal-probability stratum
        return (rng.permutation(n) + rng.random(n)) / n
    return rng.random(n)


def lognormal_mult(rng, cv, size):
    """Positive multiplicative noise with mean 1 and coefficient of variation cv."""
    s2 = np.log(1 + cv ** 2)
    return np.exp(rng.normal(-s2 / 2, np.sqrt(s2), size))


def generate(n=N_SCEN, seed=SEED, vary_season=True, vary_pmf=True):
    """n scenarios. Growth draws are identical across versions (same seed)."""
    rng = np.random.default_rng(seed)
    g_m = tri_inv(draw_u(rng, n), *MKT_G)
    g_s = tri_inv(draw_u(rng, n), *SHARE_G)

    r_w, r_d, r_p = (np.random.default_rng([seed, i]) for i in (1, 2, 3))
    wk = base_wk * lognormal_mult(r_w, CV_WEEK, (n, W)) if vary_season else np.tile(base_wk, (n, 1))
    dw = base_dow * lognormal_mult(r_d, CV_DOW, (n, D7)) if vary_season else np.tile(base_dow, (n, 1))
    pm = base_pmf * lognormal_mult(r_p, CV_PMF, (n, len(base_pmf))) if vary_pmf else np.tile(base_pmf, (n, 1))
    D_mkt = BASE_MARKET * (1 + g_m)
    return dict(g_m=g_m, g_s=g_s, D_mkt=D_mkt,
                D_tsu=D_mkt * BASE_SHARE * (1 + g_s),
                wk=wk / wk.sum(1, keepdims=True),           # renormalise
                dow=dw / dw.sum(1, keepdims=True),
                pmf=pm / pm.sum(1, keepdims=True))


def daily_market(sc):
    """(n, T) national daily market demand; day t = week*7 + dow."""
    return (sc["D_mkt"][:, None, None] * sc["wk"][:, :, None]
            * sc["dow"][:, None, :]).reshape(len(sc["D_mkt"]), -1)


# ----------------------------------------------------------------------
# 3. Summary statistics: mean, mode, sigma, X%-min / X%-max, d_bar = d + z*sigma
# ----------------------------------------------------------------------
def kde_mode(x):
    x = np.asarray(x, float)
    if np.ptp(x) < 1e-12:
        return x[0]
    grid = np.linspace(x.min(), x.max(), 512)
    return grid[np.argmax(gaussian_kde(x)(grid))]


def summarize(x):
    x = np.asarray(x, float)
    m, s = x.mean(), x.std(ddof=1)
    row = {"mean": m, "mode(KDE)": kde_mode(x), "std": s, "sample_min": x.min(),
           "sample_max": x.max()}
    for name, z in Z_LEVELS.items():
        row[f"{name} min"] = max(m - z * s, 0.0)
        row[f"{name} max"] = m + z * s              # = r-robust bound d_bar
    return row


pd.set_option("display.float_format", lambda v: f"{v:,.1f}")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

# ======================================================================
# 4. Generate the scenarios
# ======================================================================
sc = generate()
print("\n=== Scenario inputs (first 5 of", N_SCEN, ") ===")
print(pd.DataFrame({"g_market": sc["g_m"], "g_share": sc["g_s"],
                    "D_market": sc["D_mkt"], "D_tsukumo": sc["D_tsu"]}).head(5))
pd.DataFrame({"scenario": range(1, N_SCEN+1), "g_market": sc["g_m"], "g_share": sc["g_s"],
              "D_market": sc["D_mkt"], "D_tsukumo": sc["D_tsu"]}
            ).to_csv(os.path.join(BASE_DIR, "task2_scenario_inputs.csv"), index=False)

# ======================================================================
# 5. Y-A-# and Y-A-$  (annual, national)
# ======================================================================
ya = {"Market  Y-A-# (units)": sc["D_mkt"],
      "Market  Y-A-$ (dollars)": sc["D_mkt"] * PRICE,
      "Tsukumo Y-A-# (units)": sc["D_tsu"],
      "Tsukumo Y-A-$ (dollars)": sc["D_tsu"] * PRICE}
ya_tbl = pd.DataFrame({k: summarize(v) for k, v in ya.items()}).T
print("\n=== Y-A: annual national demand distribution ===")
print(ya_tbl)
ya_tbl.to_csv(os.path.join(BASE_DIR, "task2_YA_summary.csv"))

# sanity check against the triangular theory (market growth only)
a, c, b = MKT_G
th_mean = BASE_MARKET * (1 + (a + b + c) / 3)
th_std = BASE_MARKET * np.sqrt((a*a + b*b + c*c - a*b - a*c - b*c) / 18)
print(f"\nTheory (market): mean={th_mean:,.0f}  mode={BASE_MARKET*(1+c):,.0f}  std={th_std:,.0f}")
print(f"Sample (market): mean={sc['D_mkt'].mean():,.0f}  std={sc['D_mkt'].std(ddof=1):,.0f}")
print("Variability in Y-A: market growth (market) | market growth + share growth (Tsukumo);\n"
      "  week/day seasonality and ZIP3 shares only re-allocate demand, so they cancel in the annual national total.")

# ======================================================================
# 6. D-D-#  (daily demand by ZIP3)
# ======================================================================
dm = daily_market(sc)                                           # (n, T)
assert np.allclose(dm.sum(1), sc["D_mkt"])                      # seasonality preserves annual total

rank = np.argsort(-base_pmf)[:TOP_ZIP]                          # top ZIP3 by base share
top_ids = zip_ids[rank]
dd = dm[:, :, None] * sc["pmf"][:, None, rank]                  # (n, T, TOP_ZIP)

rows = []
for j, zi in enumerate(rank):
    x = dd[:, :, j]                                             # (n, T)
    t_peak = int(np.argmax(x.mean(0)))
    cells = {"annual": x.sum(1), "avg day": x.mean(1), "peak day": x[:, t_peak]}
    for cell, vals in cells.items():
        r = summarize(vals)
        rows.append({"zip3": zip_ids[zi], "state": zinfo["State"][zi], "cell": cell,
                     "week": t_peak // D7 + 1 if cell == "peak day" else np.nan,
                     "dow": t_peak % D7 + 1 if cell == "peak day" else np.nan,
                     "cv": r["std"] / r["mean"], **r})
dd_tbl = pd.DataFrame(rows)
dd_tbl.to_csv(os.path.join(BASE_DIR, "task2_DD_top_zip3_summary.csv"), index=False)
show = ["zip3", "state", "cell", "week", "dow", "mean", "std", "cv",
        "95% (1.65σ) max", "99% (2.33σ) max"]
print(f"\n=== D-D-#: top {TOP_ZIP} ZIP3 (units) ===")
print(dd_tbl[show].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

# ---------- which sources does each bound account for? (A / B / C versions) ----------
def avg_cv_day(sc_, zi):
    x = daily_market(sc_)[:, :, None] * sc_["pmf"][:, None, [zi]]
    x = x[:, :, 0]
    return float(np.mean(x.std(0, ddof=1) / x.mean(0)))


def avg_cv_natl(sc_):
    x = daily_market(sc_)
    return float(np.mean(x.std(0, ddof=1) / x.mean(0)))


versions = {"A: growth only": generate(vary_season=False, vary_pmf=False),
            "B: + week/day seasonality": generate(vary_season=True, vary_pmf=False),
            "C: + ZIP3 PMF (all)": generate(vary_season=True, vary_pmf=True)}
top1 = rank[0]
decomp = pd.DataFrame({
    "Y-A national annual": {k: v["D_mkt"].std(ddof=1) / v["D_mkt"].mean() for k, v in versions.items()},
    "D-A national daily": {k: avg_cv_natl(v) for k, v in versions.items()},
    f"D-D ZIP3 {zip_ids[top1]} daily": {k: avg_cv_day(v, top1) for k, v in versions.items()},
})
print("\n=== Average coefficient of variation (sigma/mean) by variability sources ===")
print((decomp * 100).round(1).astype(str) + "%")
decomp.to_csv(os.path.join(BASE_DIR, "task2_variability_sources_cv.csv"))

# ======================================================================
# 7. Figures
# ======================================================================
blues = ["#D6EAF8", "#85C1E9", "#2E86C1"]
line_styles = {"68% (1σ)": ":", "95% (1.65σ)": "--", "99% (2.33σ)": "-."}


def plot_dist(ax, x, title, scale, unit, theory=None):
    x = np.asarray(x, float)
    s = summarize(x)
    ax.hist(x / scale, bins=8, density=True, color="#AED6F1", edgecolor="white")
    ax.plot(x / scale, np.zeros_like(x) - 0.02 * ax.get_ylim()[1], "|", color="#1B4F72", ms=10)
    if theory is not None:
        xs = np.linspace(x.min() * 0.98, x.max() * 1.02, 300)
        ax.plot(xs / scale, theory(xs) * scale, color="#1B4F72", lw=1.5, label="Triangular (theory)")
    ax.axvline(s["mean"] / scale, color="black", lw=2, label=f"mean {s['mean']/scale:,.3f}{unit}")
    ax.axvline(s["mode(KDE)"] / scale, color="green", lw=1.5, label=f"mode {s['mode(KDE)']/scale:,.3f}{unit}")
    for name, ls in line_styles.items():
        v = s[f"{name} max"] / scale
        ax.axvline(v, color="#C0392B", ls=ls, lw=1.5, label=f"{name.split()[0]} max {v:,.3f}{unit}")
    ax.set_title(title, fontsize=10.5)
    ax.set_yticks([])
    ax.legend(fontsize=7.5, loc="upper left")
    ax.spines[["top", "right", "left"]].set_visible(False)


# Figure 1: Y-A distributions
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
th = lambda xs: tri_pdf(xs, BASE_MARKET * (1 + MKT_G[0]), BASE_MARKET * (1 + MKT_G[1]),
                        BASE_MARKET * (1 + MKT_G[2]))
plot_dist(axes[0, 0], sc["D_mkt"], "Market Y-A-#  annual demand (M units)", 1e6, "M", th)
plot_dist(axes[0, 1], sc["D_mkt"] * PRICE, "Market Y-A-$  annual demand ($B)", 1e9, "B",
          lambda xs: tri_pdf(xs / PRICE, BASE_MARKET * (1 + MKT_G[0]), BASE_MARKET * (1 + MKT_G[1]),
                             BASE_MARKET * (1 + MKT_G[2])) / PRICE)
plot_dist(axes[1, 0], sc["D_tsu"], "Tsukumo Y-A-#  annual demand (K units)", 1e3, "K")
plot_dist(axes[1, 1], sc["D_tsu"] * PRICE, "Tsukumo Y-A-$  annual demand ($M)", 1e6, "M")
fig.suptitle(f"Y-A: distribution over {N_SCEN} scenarios (growth uncertainty only)",
             fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "t2_fig1_YA_distributions.png"), dpi=200)

# Figure 2: D-D fan chart (top ZIP3) + variability sources
fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1])
ax = fig.add_subplot(gs[0])
x = dd[:, :, 0]
mu, sd = x.mean(0), x.std(0, ddof=1)
wk_axis = np.arange(T) / D7 + 1
for k in range(N_SCEN):
    ax.plot(wk_axis, x[k], color="gray", lw=0.4, alpha=0.35)
for (name, z), col in zip(reversed(list(Z_LEVELS.items())), blues[::-1][::-1]):
    ax.fill_between(wk_axis, np.maximum(mu - z * sd, 0), mu + z * sd, color=col, alpha=0.55,
                    label=f"±{name}")
ax.plot(wk_axis, mu, color="#1B4F72", lw=1.8, label="mean")
ax.set_xlabel("Week of year (7 days per week)")
ax.set_ylabel("Units per day")
ax.set_title(f"D-D-#  Daily demand, ZIP3 {zip_ids[top1]} ({zinfo['State'][top1]}): "
             f"{N_SCEN} scenarios with 68/95/99% bands", fontweight="bold")
ax.legend(loc="upper right", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[1])
xg = np.arange(len(decomp.columns))
wbar = 0.26
cols = ["#AED6F1", "#3498DB", "#1B4F72"]
for i, (vname, col) in enumerate(zip(decomp.index, cols)):
    bars = ax.bar(xg + (i - 1) * wbar, decomp.loc[vname].values * 100, wbar, color=col, label=vname)
    for bb in bars:
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.3,
                f"{bb.get_height():.1f}%", ha="center", fontsize=8)
ax.set_xticks(xg, [c.replace(" ", "\n", 1) for c in decomp.columns], fontsize=8.5)
ax.set_ylabel("Average CV (σ / mean, %)")
ax.set_title("Variability by source", fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "t2_fig2_DD_fan_and_sources.png"), dpi=200)

# ======================================================================
# 8. Y-B-#  : annual demand by market type (+ stability of the Primary share)
# ======================================================================
TYPES = ["Primary", "Secondary", "Tertiary"]
mtype = zinfo["Market"].str.strip().str.capitalize().values
type_idx = {t: np.where(mtype == t)[0] for t in TYPES}
type_share = np.column_stack([sc["pmf"][:, type_idx[t]].sum(1) for t in TYPES])      # (n, 3)
base_type_share = np.array([base_pmf[type_idx[t]].sum() for t in TYPES])
type_units = sc["D_mkt"][:, None] * type_share                                          # (n, 3)
type_colors = {"Primary": "#1B4F72", "Secondary": "#5DADE2", "Tertiary": "#F5B041"}


def bars_with_bounds(ax, labels, samples, colors, horizontal=False):
    """samples (n, m): bar = mean; thick whisker = 95% (1.65sigma); thin whisker = 99% (2.33sigma)."""
    mu, sd = samples.mean(0), samples.std(0, ddof=1)
    pos = np.arange(len(labels))
    for z, lw in ((2.33, 1.0), (1.65, 3.0)):
        lo, hi = np.minimum(z * sd, mu), z * sd
        kw = dict(fmt="none", ecolor="#C0392B", elinewidth=lw, capsize=6 if z == 2.33 else 0)
        if horizontal:
            ax.errorbar(mu, pos, xerr=[lo, hi], **kw)
        else:
            ax.errorbar(pos, mu, yerr=[lo, hi], **kw)
    if horizontal:
        ax.barh(pos, mu, color=colors, alpha=0.9, zorder=0)
        ax.set_yticks(pos, labels)
        ax.invert_yaxis()
    else:
        ax.bar(pos, mu, color=colors, alpha=0.9, zorder=0)
        ax.set_xticks(pos, labels)
    return mu, sd


yb_units = pd.DataFrame({t: summarize(type_units[:, i]) for i, t in enumerate(TYPES)}).T
yb_share = pd.DataFrame({t: summarize(type_share[:, i] * 100) for i, t in enumerate(TYPES)}).T
yb_share.insert(0, "base_share_%", base_type_share * 100)
print("\n=== Y-B-#: annual demand by market type (units) ===")
print(yb_units[["mean", "std", "68% (1σ) max", "95% (1.65σ) max", "99% (2.33σ) max"]])
print("\n=== Y-B: market-type SHARE (%) across scenarios ===")
print(yb_share[["base_share_%", "mean", "std", "sample_min", "sample_max",
                "95% (1.65σ) min", "95% (1.65σ) max"]].round(2))
yb_units.to_csv(os.path.join(BASE_DIR, "task2_YB_units_summary.csv"))
yb_share.to_csv(os.path.join(BASE_DIR, "task2_YB_share_summary.csv"))
print("Variability in Y-B: market growth + ZIP3 PMF (week/day seasonality cancels in the annual total).")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
mu, sd = bars_with_bounds(ax, TYPES, type_units, [type_colors[t] for t in TYPES])
for i, (m_, s_) in enumerate(zip(mu, sd)):
    ax.text(i, m_ + 2.4 * s_ + mu.max() * 0.01, f"{m_/1e3:,.0f}K\n99% max {(m_+2.33*s_)/1e3:,.0f}K",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Annual demand (units)")
ax.set_ylim(0, mu.max() * 1.18)
ax.set_title("Y-B-#  Annual demand by market type\n(bar = mean; whiskers = 95% thick / 99% thin)")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
rng_j = np.random.default_rng(0)
for i, t in enumerate(TYPES):
    dev = (type_share[:, i] - base_type_share[i]) * 100
    ax.scatter(i + rng_j.uniform(-0.15, 0.15, len(dev)), dev, color=type_colors[t], alpha=0.8,
               edgecolor="white", zorder=3)
    s_ = dev.std(ddof=1)
    ax.plot([i - 0.3, i + 0.3], [1.65 * s_] * 2, color="#C0392B", ls="--")
    ax.plot([i - 0.3, i + 0.3], [-1.65 * s_] * 2, color="#C0392B", ls="--")
    ax.text(i + 0.32, 1.65 * s_, f"±{1.65 * s_:.2f} pp", va="center", fontsize=8.5, color="#C0392B")
ax.axhline(0, color="black", lw=1)
ax.set_xticks(range(3), [f"{t}\nbase {b*100:.1f}%" for t, b in zip(TYPES, base_type_share)])
ax.set_xlim(-0.5, 2.9)
ax.set_ylabel("Scenario share − baseline share (percentage points)")
ax.set_title("Y-B  Stability of market-type shares\n(dots = 25 scenarios; dashed = ±1.65σ)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "t2_fig3_YB_market_type.png"), dpi=200)

# ======================================================================
# 9. Y-C-#  : annual demand by state (+ rank stability)
# ======================================================================
states = zinfo["State"].astype(str).to_numpy()
state_list = np.array(sorted(set(states)))
onehot = (states[:, None] == state_list[None, :]).astype(float)                 # (Z, S)
state_share = sc["pmf"] @ onehot                                                # (n, S)
state_units = sc["D_mkt"][:, None] * state_share
state_rank = (-state_units).argsort(1).argsort(1) + 1                           # rank 1 = largest

rows = []
for j, st in enumerate(state_list):
    r = summarize(state_units[:, j])
    rows.append({"state": st, "cv": r["std"] / r["mean"], **r,
                 "rank_mean": state_rank[:, j].mean(), "rank_best": state_rank[:, j].min(),
                 "rank_worst": state_rank[:, j].max()})
yc = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
yc.to_csv(os.path.join(BASE_DIR, "task2_YC_states_summary.csv"), index=False)
print("\n=== Y-C-#: top 15 states (units) ===")
print(yc.head(15)[["state", "mean", "std", "cv", "95% (1.65σ) max", "99% (2.33σ) max",
                   "rank_best", "rank_worst"]].to_string(index=False,
      float_format=lambda v: f"{v:,.2f}" if abs(v) < 10 else f"{v:,.0f}"))
print("Variability in Y-C: market growth + ZIP3 PMF.")

TOP_S = 15
top_st = yc.head(TOP_S)
j_top = [int(np.where(state_list == s)[0][0]) for s in top_st["state"]]
fig, ax = plt.subplots(figsize=(13, 5.8))
mu, sd = bars_with_bounds(ax, list(top_st["state"]), state_units[:, j_top], "#2E86C1")
for i, j in enumerate(j_top):
    ax.text(i, mu[i] + 2.4 * sd[i] + mu.max() * 0.01,
            f"rank\n{state_rank[:, j].min()}–{state_rank[:, j].max()}", ha="center",
            va="bottom", fontsize=8)
ax.set_ylabel("Annual demand (units)")
ax.set_ylim(0, (mu + 2.33 * sd).max() * 1.15)
ax.set_title(f"Y-C-#  Top {TOP_S} states: annual demand with robust bounds\n"
             "(bar = mean; whiskers = 95% thick / 99% thin; text = rank range over 25 scenarios)",
             fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "t2_fig4_YC_states.png"), dpi=200)

# ======================================================================
# 10. Y-D-#  : annual demand by ZIP3 (+ Pareto concentration stability)
# ======================================================================
zip_units = sc["D_mkt"][:, None] * sc["pmf"]                                    # (n, Z)
yd = pd.DataFrame([{"zip3": zip_ids[z], "state": zinfo["State"][z],
                    "market": zinfo["Market"][z], **summarize(zip_units[:, z])}
                   for z in range(len(zip_ids))])
yd["cv"] = yd["std"] / yd["mean"]
yd = yd.sort_values("mean", ascending=False).reset_index(drop=True)
yd.to_csv(os.path.join(BASE_DIR, "task2_YD_zip3_summary.csv"), index=False)
print("\n=== Y-D-#: top 15 ZIP3 (units) ===")
print(yd.head(15)[["zip3", "state", "mean", "std", "cv", "95% (1.65σ) max",
                   "99% (2.33σ) max"]].to_string(index=False,
      float_format=lambda v: f"{v:,.2f}" if abs(v) < 10 else f"{v:,.0f}"))
print("Variability in Y-D: market growth + ZIP3 PMF.")


def n_cover(shares, q=0.8):
    c = np.cumsum(np.sort(shares)[::-1])
    return int(np.searchsorted(c, q) + 1)


n80_zip = np.array([n_cover(sc["pmf"][k]) for k in range(N_SCEN)])
n80_state = np.array([n_cover(state_share[k]) for k in range(N_SCEN)])
n80_tbl = pd.DataFrame({"ZIP3 to cover 80%": summarize(n80_zip),
                        "States to cover 80%": summarize(n80_state)}).T
print("\n=== Pareto stability: number of areas needed to cover 80% of demand ===")
print(n80_tbl[["mean", "std", "sample_min", "sample_max", "95% (1.65σ) max"]].round(1))
n80_tbl.to_csv(os.path.join(BASE_DIR, "task2_pareto_n80_summary.csv"))

TOP_Z15 = 15
topz = yd.head(TOP_Z15)
jz = [int(np.where(zip_ids == z)[0][0]) for z in topz["zip3"]]
fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
zc = {t: type_colors.get(t.capitalize(), "gray") for t in topz["market"].unique()}
mu, sd = bars_with_bounds(ax, [f"{z} ({s})" for z, s in zip(topz["zip3"], topz["state"])],
                          zip_units[:, jz], [zc[m] for m in topz["market"]], horizontal=True)
for i, (m_, s_) in enumerate(zip(mu, sd)):
    ax.text(m_ + 2.4 * s_ + mu.max() * 0.01, i, f"{m_:,.0f}  (99% max {m_ + 2.33 * s_:,.0f})",
            va="center", fontsize=8)
ax.set_xlim(0, (mu + 2.33 * sd).max() * 1.45)
ax.set_xlabel("Annual demand (units)")
ax.set_title(f"Y-D-#  Top {TOP_Z15} ZIP3s with robust bounds", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
xp = np.arange(1, len(base_pmf) + 1) / len(base_pmf) * 100
curves = np.array([np.cumsum(np.sort(sc["pmf"][k])[::-1]) * 100 for k in range(N_SCEN)])
for k in range(N_SCEN):
    ax.plot(xp, curves[k], color="gray", lw=0.6, alpha=0.5)
ax.plot(xp, curves.mean(0), color="#1B4F72", lw=2.5, label="mean of 25 scenarios")
ax.axhline(80, color="#C0392B", ls=":")
ax.set_xlabel("% of ZIP3 areas (largest first)")
ax.set_ylabel("Cumulative % of demand")
ax.set_title(f"Pareto stability: ZIP3s covering 80% of demand = "
             f"{n80_zip.mean():.0f} ± {n80_zip.std(ddof=1):.1f}", fontweight="bold")
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "t2_fig5_YD_zip3_pareto.png"), dpi=200)

# ======================================================================
# 11. D-B-#  : daily demand by market type
# ======================================================================
type_daily = dm[:, :, None] * type_share[:, None, :]                            # (n, T, 3)
rows, daily_rows = [], []
for i, t in enumerate(TYPES):
    x = type_daily[:, :, i]
    t_peak = int(np.argmax(x.mean(0)))
    cells = {"avg day": x.mean(1), "peak day (fixed calendar day)": x[:, t_peak],
             "annual max day (per scenario)": x.max(1)}
    avg_mean = x.mean()
    for cell, vals in cells.items():
        r = summarize(vals)
        rows.append({"market": t, "cell": cell, "cv": r["std"] / r["mean"],
                     "ratio_to_avg_day": r["mean"] / avg_mean, **r})
    mu_t, sd_t = x.mean(0), x.std(0, ddof=1)
    daily_rows.append(pd.DataFrame({"market": t, "week": np.arange(T) // D7 + 1,
                                    "dow": np.arange(T) % D7 + 1, "mean": mu_t, "std": sd_t,
                                    "68% max": mu_t + sd_t, "95% max": mu_t + 1.65 * sd_t,
                                    "99% max": mu_t + 2.33 * sd_t}))
db = pd.DataFrame(rows)
db.to_csv(os.path.join(BASE_DIR, "task2_DB_summary.csv"), index=False)
pd.concat(daily_rows).to_csv(os.path.join(BASE_DIR, "task2_DB_daily_bounds.csv"), index=False)
print("\n=== D-B-#: daily demand by market type (units/day) ===")
print(db[["market", "cell", "mean", "std", "cv", "ratio_to_avg_day", "95% (1.65σ) max",
          "99% (2.33σ) max"]].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
print("Variability in D-B: market growth + week + day-of-week + ZIP3 PMF (through the type share).")

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for i, (ax, t) in enumerate(zip(axes, TYPES)):
    x = type_daily[:, :, i]
    mu_t, sd_t = x.mean(0), x.std(0, ddof=1)
    for (name, z), col in zip(reversed(list(Z_LEVELS.items())), blues):
        ax.fill_between(wk_axis, np.maximum(mu_t - z * sd_t, 0), mu_t + z * sd_t, color=col,
                        alpha=0.6, label=f"±{name}")
    ax.plot(wk_axis, mu_t, color="#1B4F72", lw=1.3, label="mean")
    ax.set_ylabel("Units / day")
    ax.set_title(f"{t}: avg day {x.mean():,.0f} | 99% peak bound {(mu_t + 2.33 * sd_t).max():,.0f}",
                 fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(ncol=4, fontsize=8, loc="upper right")
axes[-1].set_xlabel("Week of year")
fig.suptitle("D-B-#  Daily demand by market type: 25 scenarios, 68/95/99% bands",
             fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "t2_fig6_DB_fan.png"), dpi=200)

fig, ax = plt.subplots(figsize=(11, 5.8))
cell_names = list(db["cell"].unique())
cell_cols = ["#AED6F1", "#3498DB", "#1B4F72"]
wb = 0.26
for c_i, (cn, col) in enumerate(zip(cell_names, cell_cols)):
    sub = db[db["cell"] == cn].set_index("market").loc[TYPES]
    xpos = np.arange(3) + (c_i - 1) * wb
    ax.bar(xpos, sub["mean"], wb, color=col, label=cn)
    ax.errorbar(xpos, sub["mean"], yerr=[np.minimum(1.65 * sub["std"], sub["mean"]), 1.65 * sub["std"]],
                fmt="none", ecolor="#C0392B", elinewidth=3)
    ax.errorbar(xpos, sub["mean"], yerr=[np.minimum(2.33 * sub["std"], sub["mean"]), 2.33 * sub["std"]],
                fmt="none", ecolor="#C0392B", elinewidth=1, capsize=5)
    for xp_, m_, s_ in zip(xpos, sub["mean"], sub["std"]):
        ax.text(xp_, m_ + 2.4 * s_ + sub["mean"].max() * 0.01, f"{m_ + 2.33 * s_:,.0f}",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(3), TYPES)
ax.set_ylabel("Units per day")
ax.set_title("D-B-#  Daily range by market type\n(bar = mean; whiskers = 95% thick / 99% thin; "
             "number = 99% bound)", fontweight="bold")
ax.legend(fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "t2_fig7_DB_range.png"), dpi=200)

# ======================================================================
# 12. D-D-#  : local peak days (top ZIP3s)
# ======================================================================
amax = dd.max(axis=1)                                                           # (n, TOP_ZIP)
davg = dd.mean(axis=1)
rows = []
for j, zi in enumerate(rank):
    r = summarize(amax[:, j])
    rows.append({"zip3": zip_ids[zi], "state": zinfo["State"][zi], "avg_day_mean": davg[:, j].mean(),
                 "max_day_mean": r["mean"], "max_day_std": r["std"],
                 "peak_to_avg": r["mean"] / davg[:, j].mean(),
                 "max_day_95%": r["95% (1.65σ) max"], "max_day_99%": r["99% (2.33σ) max"]})
ddm = pd.DataFrame(rows)
ddm.to_csv(os.path.join(BASE_DIR, "task2_DD_annual_max_day.csv"), index=False)
print("\n=== D-D-#: annual maximum day per scenario, top ZIP3 (units/day) ===")
print(ddm.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

fig, ax = plt.subplots(figsize=(12, 6))
labels = [f"{z} ({s})" for z, s in zip(ddm["zip3"], ddm["state"])]
pos = np.arange(len(labels))
ax.bar(pos - 0.2, ddm["avg_day_mean"], 0.38, color="#AED6F1", label="average day (mean)")
ax.bar(pos + 0.2, ddm["max_day_mean"], 0.38, color="#1B4F72", label="annual max day (mean)")
sd_m = ddm["max_day_std"].values
ax.errorbar(pos + 0.2, ddm["max_day_mean"], yerr=[np.minimum(1.65 * sd_m, ddm["max_day_mean"]), 1.65 * sd_m],
            fmt="none", ecolor="#C0392B", elinewidth=3, label="95% bound")
ax.errorbar(pos + 0.2, ddm["max_day_mean"], yerr=[np.minimum(2.33 * sd_m, ddm["max_day_mean"]), 2.33 * sd_m],
            fmt="none", ecolor="#C0392B", elinewidth=1, capsize=5, label="99% bound")
for p_, m_, s_, r_ in zip(pos, ddm["max_day_mean"], sd_m, ddm["peak_to_avg"]):
    ax.text(p_ + 0.2, m_ + 2.4 * s_ + ddm["max_day_mean"].max() * 0.01, f"×{r_:.1f}", ha="center",
            fontsize=8.5)
ax.set_xticks(pos, labels, rotation=30, ha="right")
ax.set_ylabel("Units per day")
ax.set_title("D-D-#  Local peak: annual max day vs. average day, top ZIP3s\n"
             "(×N = peak-to-average ratio)", fontweight="bold")
ax.legend(fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "t2_fig8_DD_peak_days.png"), dpi=200)

# ======================================================================
# 14. MASTER REPORT: r-robust demand bounds  d_bar = d + z_r * sigma
#     One row per reported cell, for every combination (Y-A, Y-B, Y-C, Y-D,
#     D-B, D-D). Requires this file to be appended AFTER:
#       task2_uncertainty.py  (defines ya_tbl)
#       task2_addon_YB_YC_YD_DB_DD.py  (defines yb_units, yc, yd, db, ddm)
#     Produces: task2_master_report.csv  +  a printed / figure summary.
# ======================================================================

# ---------- helper: pull the 6 required items out of a summarize() row ----------
def six_items(row, combination, cell, sources):
    return {
        "Combination": combination,
        "Cell": cell,
        "d = mean": row["mean"],
        "sigma = std": row["std"],
        "68% (z=1.00) bound": row["68% (1σ) max"],
        "95% (z=1.65) bound": row["95% (1.65σ) max"],
        "99% (z=2.33) bound": row["99% (2.33σ) max"],
        "Variability sources covered": sources,
    }


SRC_GROWTH_ONLY = "Market growth only"
SRC_GROWTH_SHARE = "Market growth + Tsukumo share growth"
SRC_GROWTH_PMF = "Market growth + ZIP3 demand-share (PMF) uncertainty"
SRC_ALL_TYPE = ("Market growth + week-of-year + day-of-week seasonality "
                "+ ZIP3 PMF uncertainty (via market-type share)")
SRC_ALL_ZIP = ("Market growth + week-of-year + day-of-week seasonality "
              "+ ZIP3 PMF uncertainty (all four sources)")

rows = []

# ---- Y-A-# / Y-A-$ (market and Tsukumo) ----
for label, src in [("Market  Y-A-# (units)", SRC_GROWTH_ONLY),
                    ("Market  Y-A-$ (dollars)", SRC_GROWTH_ONLY),
                    ("Tsukumo Y-A-# (units)", SRC_GROWTH_SHARE),
                    ("Tsukumo Y-A-$ (dollars)", SRC_GROWTH_SHARE)]:
    rows.append(six_items(ya_tbl.loc[label], "Y-A", label, src))

# ---- Y-B-# (3 market types) ----
for t in TYPES:
    rows.append(six_items(yb_units.loc[t], "Y-B", f"{t} (annual units)", SRC_GROWTH_PMF))

# ---- Y-C-# (top 10 states) ----
TOP_C = 10
for _, r in yc.head(TOP_C).iterrows():
    rows.append(six_items(r, "Y-C", f"{r['state']} (annual units)", SRC_GROWTH_PMF))

# ---- Y-D-# (top 10 ZIP3) ----
TOP_D = 10
for _, r in yd.head(TOP_D).iterrows():
    rows.append(six_items(r, "Y-D", f"ZIP3 {r['zip3']} ({r['state']}, annual units)", SRC_GROWTH_PMF))

# ---- D-B-# (3 market types x avg-day / annual-max-day) ----
for cellname, src in [("avg day", SRC_ALL_TYPE), ("annual max day (per scenario)", SRC_ALL_TYPE)]:
    for t in TYPES:
        r = db[(db["market"] == t) & (db["cell"] == cellname)].iloc[0]
        rows.append(six_items(r, "D-B", f"{t} – {cellname}", src))

# ---- D-D-# (top 10 ZIP3, avg day vs annual max day) ----
TOP_DD = 10
for _, r in ddm.head(TOP_DD).iterrows():
    rows.append(six_items({"mean": r["avg_day_mean"], "std": np.nan,
                           "68% (1σ) max": np.nan, "95% (1.65σ) max": np.nan,
                           "99% (2.33σ) max": np.nan},
                          "D-D", f"ZIP3 {r['zip3']} ({r['state']}) – avg day", "n/a (reference row)"))
    rows.append(six_items({"mean": r["max_day_mean"], "std": r["max_day_std"],
                           "68% (1σ) max": r["max_day_mean"] + r["max_day_std"],
                           "95% (1.65σ) max": r["max_day_95%"],
                           "99% (2.33σ) max": r["max_day_99%"]},
                          "D-D", f"ZIP3 {r['zip3']} ({r['state']}) – annual max day", SRC_ALL_ZIP))

master = pd.DataFrame(rows)
master.to_csv(os.path.join(BASE_DIR, "task2_master_report.csv"), index=False)

pd.set_option("display.max_rows", 200)
print(f"\n=== MASTER REPORT: {len(master)} reported cells ===")
print(master.to_string(index=False, float_format=lambda v: f"{v:,.1f}" if pd.notna(v) else "n/a"))

# ======================================================================
# 15. Figures: one table image per combination (easy to paste into a report)
# ======================================================================
import textwrap


def render_table(df_, title, fname, fontsize=9.5):
    uniform_src = df_["Variability sources covered"].nunique() == 1
    show_cols = ["Cell", "d = mean", "sigma = std", "68% (z=1.00) bound",
                "95% (z=1.65) bound", "99% (z=2.33) bound"]
    if not uniform_src:
        show_cols = show_cols + ["Variability sources covered"]

    def fmt(c, col):
        if isinstance(c, str):
            return "\n".join(textwrap.wrap(c, 26)) if col == "Variability sources covered" else c
        return "n/a" if pd.isna(c) else f"{c:,.0f}"

    cell_text = [[fmt(c, col) for c, col in zip(row, show_cols)] for row in df_[show_cols].values]
    n_wrap_lines = max((v.count("\n") + 1 for row in cell_text for v in row), default=1)
    header_labels = ["\n".join(textwrap.wrap(c, 14)) if c != "Cell" else c for c in show_cols]
    header_lines = max(h.count("\n") + 1 for h in header_labels)
    fig_h = 0.35 + 0.20 * header_lines + (0.34 * max(n_wrap_lines, 1)) * len(df_)
    fig_w = 14.5 if uniform_src else 18.5
    max_cell_len = max(len(str(v)) for v in df_["Cell"])
    cell_w = min(0.18 + 0.010 * max_cell_len, 0.34)
    if uniform_src:
        other_w = (1 - cell_w) / 5
        col_widths = [cell_w] + [other_w] * 5
    else:
        src_w = 0.22
        num_w = (1 - cell_w - src_w) / 5
        col_widths = [cell_w] + [num_w] * 5 + [src_w]

    title_lines = 2
    title_h = 0.42 * title_lines
    fig_h_total = fig_h + title_h
    table_frac = fig_h / fig_h_total          # fraction of the figure the table occupies

    fig, ax = plt.subplots(figsize=(fig_w, fig_h_total))
    ax.axis("off")
    t = ax.table(cellText=cell_text, colLabels=header_labels, cellLoc="center",
                 colWidths=col_widths, bbox=[0, 0, 1, table_frac])
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, 1.5 * max(n_wrap_lines, header_lines))
    last_col = len(show_cols) - 1
    for (r, c), cellobj in t.get_celld().items():
        if r == 0:
            cellobj.set_facecolor("#1B4F72")
            cellobj.set_text_props(color="white", fontweight="bold")
        if c == 0 or (not uniform_src and c == last_col):
            cellobj.set_text_props(ha="left")
        else:
            cellobj.set_text_props(ha="right")
    src_note = df_["Variability sources covered"].iloc[0] if uniform_src else "shown per row (right column)"
    ax.set_title(f"{title}\nVariability sources: {src_note}", fontweight="bold",
                 fontsize=11, y=table_frac + 0.02, loc="center")
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=200, bbox_inches="tight")


render_table(master[master["Combination"] == "Y-A"],
             "Y-A-# / Y-A-$  – r-robust demand bounds", "t2_report_YA.png")
render_table(master[master["Combination"] == "Y-B"],
             "Y-B-#  – r-robust demand bounds by market type", "t2_report_YB.png")
render_table(master[master["Combination"] == "Y-C"],
             f"Y-C-#  – r-robust demand bounds, top {TOP_C} states", "t2_report_YC.png")
render_table(master[master["Combination"] == "Y-D"],
             f"Y-D-#  – r-robust demand bounds, top {TOP_D} ZIP3", "t2_report_YD.png")
render_table(master[(master["Combination"] == "D-B")],
             "D-B-#  – r-robust demand bounds by market type (avg day / annual max day)",
             "t2_report_DB.png", fontsize=9)
render_table(master[(master["Combination"] == "D-D") & master["Cell"].str.contains("annual max day")],
             f"D-D-#  – r-robust demand bounds, top {TOP_DD} ZIP3 (annual max day)",
             "t2_report_DD.png")

print("\nSaved: task2_master_report.csv  and one table image per combination in /plot")