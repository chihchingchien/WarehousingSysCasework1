"""
Task 1 - Overall Scale (Y-A-# and Y-A-$, plus V and W)
No uncertainty, no seasonality.
Notation: d_mt = total market demand, ds_cmt = Tsukumo's captured demand.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR                                  # CSVs live next to this script
PLOT_DIR = os.path.join(BASE_DIR, "plot")            # figures go here
os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Parameters
# ----------------------------------------------------------------------
BASE_MARKET = 2_000_000          # current annual US market (units)
MARKET_GROWTH = {"Conservative": 0.04, "Expected": 0.075, "Optimistic": 0.12}

BASE_SHARE = 0.036               # Tsukumo current share
SHARE_GROWTH = {"Min (15%)": 0.15, "Median (20%)": 0.20, "Max (25%)": 0.25}

PRICE, WEIGHT, VOLUME = 3_000, 60, 3 * 2 * 2   # $/unit, lbs/unit, ft3/unit
DAYS, MONTHS = 365, 12

# ----------------------------------------------------------------------
# 2. Overall market demand d_mt (Y-A) per scenario
# ----------------------------------------------------------------------
rows = []
for name, g in MARKET_GROWTH.items():
    units = BASE_MARKET * (1 + g)
    rows.append({
        "Scenario": name,
        "Growth": g,
        "Units (#)": units,
        "Dollars ($)": units * PRICE,
        "Weight (lbs)": units * WEIGHT,
        "Volume (ft3)": units * VOLUME,
    })
df = pd.DataFrame(rows).set_index("Scenario")

# Uniform temporal split (no seasonality) -> for context only
df["Units / month"] = df["Units (#)"] / MONTHS
df["Units / day"] = df["Units (#)"] / DAYS
df["Volume / day (ft3)"] = df["Volume (ft3)"] / DAYS
df["Weight / day (lbs)"] = df["Weight (lbs)"] / DAYS

pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
pd.set_option("display.width", 200)
print("=== Overall market demand d_mt (Y-A) ===")
print(df.drop(columns="Growth").T)

# ----------------------------------------------------------------------
# 3. Tsukumo's captured demand ds_cmt: 3 market x 3 share-growth scenarios
# ----------------------------------------------------------------------
share_next = {k: BASE_SHARE * (1 + g) for k, g in SHARE_GROWTH.items()}
ds = pd.DataFrame(
    {sk: {mk: df.loc[mk, "Units (#)"] * s for mk in df.index}
     for sk, s in share_next.items()}
)
print("\n=== Tsukumo share next year ===")
print({k: f"{v:.3%}" for k, v in share_next.items()})
print("\n=== Tsukumo demand ds_cmt (units) ===")
print(ds)
print("\nCurrent Tsukumo demand (units):", BASE_MARKET * BASE_SHARE)

# ----------------------------------------------------------------------
# 4. Plots
# ----------------------------------------------------------------------
colors = ["#7FB3D5", "#2E86C1", "#1B4F72"]
scen = list(df.index)

def annotate(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt(b.get_height()),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

# --- Figure 1: Y-A-# and Y-A-$ side by side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
bars = ax.bar(scen, df["Units (#)"] / 1e6, color=colors)
ax.axhline(BASE_MARKET / 1e6, color="gray", ls="--", lw=1)
ax.text(2.45, BASE_MARKET / 1e6 + 0.01, "Current: 2.00M", ha="right",
        color="gray", fontsize=9)
annotate(ax, bars, lambda h: f"{h:.2f}M")
for b, g in zip(bars, df["Growth"]):
    ax.text(b.get_x() + b.get_width() / 2, 0.05, f"+{g:.1%}",
            ha="center", color="white", fontweight="bold")
ax.set_title("Y-A-#  Annual market demand (units)")
ax.set_ylabel("Million units")
ax.set_ylim(0, df["Units (#)"].max() / 1e6 * 1.12)

ax = axes[1]
bars = ax.bar(scen, df["Dollars ($)"] / 1e9, color=colors)
annotate(ax, bars, lambda h: f"${h:.2f}B")
ax.set_title("Y-A-$  Annual market demand (dollars)")
ax.set_ylabel("Billion USD")
ax.set_ylim(0, df["Dollars ($)"].max() / 1e9 * 1.12)

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Overall US market demand d_mt over 1-year horizon "
             "(Tsukumo + competitors)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig1_YA_units_dollars.png"), dpi=200)

# --- Figure 2: Y-A-W and Y-A-V ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bars = axes[0].bar(scen, df["Weight (lbs)"] / 1e6, color=colors)
annotate(axes[0], bars, lambda h: f"{h:.0f}M lbs")
axes[0].set_title("Y-A-W  Annual weight")
axes[0].set_ylabel("Million lbs")
axes[0].set_ylim(0, df["Weight (lbs)"].max() / 1e6 * 1.12)

bars = axes[1].bar(scen, df["Volume (ft3)"] / 1e6, color=colors)
annotate(axes[1], bars, lambda h: f"{h:.1f}M ft³")
axes[1].set_title("Y-A-V  Annual volume")
axes[1].set_ylabel("Million ft³")
axes[1].set_ylim(0, df["Volume (ft3)"].max() / 1e6 * 1.12)
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig2_YA_weight_volume.png"), dpi=200)

# --- Figure 3: Tsukumo demand heatmap (market growth x share growth) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.imshow(ds.values, cmap="Blues")
ax.set_xticks(range(3), [f"{k}\n(share {v:.2%})" for k, v in share_next.items()])
ax.set_yticks(range(3), [f"{s} (+{MARKET_GROWTH[s]:.1%})" for s in scen])
for i in range(3):
    for j in range(3):
        v = ds.values[i, j]
        ax.text(j, i, f"{v:,.0f}\n${v * PRICE / 1e6:,.0f}M", ha="center",
                va="center", fontsize=10,
                color="white" if v > ds.values.mean() else "black")
ax.set_xlabel("Tsukumo market-share growth scenario")
ax.set_ylabel("Market growth scenario")
ax.set_title(f"Tsukumo demand ds_cmt (units, $)\n"
             f"Current: {BASE_MARKET * BASE_SHARE:,.0f} units")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig3_tsukumo_share_heatmap.png"), dpi=200)

# --- Save summary tables ---
df.to_csv("task1_overall_scale_market.csv")
ds.to_csv("task1_overall_scale_tsukumo.csv")
# plt.show()  # (moved to end of script)


# ======================================================================
# Part 2: Y-B-# and Y-B-$  (annual demand by market type)
# ======================================================================

# ---------- 1. Load data (ZIP3 read as str to keep leading zeros) ----------
# sep=None lets pandas auto-detect comma / tab separators
zmkt = pd.read_csv(os.path.join(DATA_DIR, "zip3_market.csv"), dtype=str,
                   sep=None, engine="python", encoding="utf-8-sig")
zpmf = pd.read_csv(os.path.join(DATA_DIR, "zip3_pmf.csv"), dtype=str,
                   sep=None, engine="python", encoding="utf-8-sig")
# strip whitespace and any leftover BOM from column names
zmkt.columns = zmkt.columns.str.replace("\ufeff", "", regex=False).str.strip()
zpmf.columns = zpmf.columns.str.replace("\ufeff", "", regex=False).str.strip()
print("\nzip3_market.csv columns:", list(zmkt.columns))
print("zip3_pmf.csv columns   :", list(zpmf.columns))


def find_col(df, keywords, exclude=()):
    """Return the first column whose name contains any keyword (case-insens.)."""
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in keywords) and not any(x in low for x in exclude):
            return c
    raise KeyError(f"No column matching {keywords} in {list(df.columns)}. "
                   f"Set the column name manually below.")


# If auto-detection picks the wrong column, overwrite these 4 names manually.
# Columns: zip3_market.csv -> ZIP3, Market, State ; zip3_pmf.csv -> ZIP3, PMF
ZIP_COL_M, TYPE_COL = "ZIP3", "Market"
ZIP_COL_P, PMF_COL = "ZIP3", "PMF"

zmkt["zip3"] = zmkt[ZIP_COL_M].str.strip().str.zfill(3)
zpmf["zip3"] = zpmf[ZIP_COL_P].str.strip().str.zfill(3)
zpmf["pmf"] = pd.to_numeric(zpmf[PMF_COL])


def norm_type(s):
    s = str(s).strip().lower()
    for t in ("primary", "secondary", "tertiary"):
        if t in s:
            return t.capitalize()
    return s.capitalize()


zmkt["market_type"] = zmkt[TYPE_COL].map(norm_type)
TYPES = ["Primary", "Secondary", "Tertiary"]

# ---------- 2. Merge: ZIP3 in market file but not in PMF -> share = 0 ----------
z = zmkt[["zip3", "market_type"]].drop_duplicates("zip3").merge(
    zpmf[["zip3", "pmf"]].drop_duplicates("zip3"), on="zip3", how="left")
z["pmf"] = z["pmf"].fillna(0.0)

print("\n--- Data checks ---")
print(f"ZIP3 in market file        : {len(z)}")
print(f"ZIP3 without PMF (set 0)   : {(~z['zip3'].isin(zpmf['zip3'])).sum()}")
print(f"ZIP3 in PMF but not market : {(~zpmf['zip3'].isin(zmkt['zip3'])).sum()}")
print(f"PMF sum (matched ZIP3s)    : {z['pmf'].sum():.6f}  (PMF file total: {zpmf['pmf'].sum():.6f})")

# ---------- 3. Aggregate to market type ----------
by_type = z.groupby("market_type").agg(n_zip3=("zip3", "count"),
                                       share=("pmf", "sum")).reindex(TYPES)
by_type["zip3_pct"] = by_type["n_zip3"] / by_type["n_zip3"].sum()

# Demand per scenario: d_mt(type) = d_t(USA) * share(type)
units_bt = pd.DataFrame({s: df.loc[s, "Units (#)"] * by_type["share"]
                         for s in scen})                       # rows = type
dollars_bt = units_bt * PRICE
by_type["units_expected"] = units_bt["Expected"]
by_type["dollars_expected"] = dollars_bt["Expected"]
by_type["units_per_zip3"] = by_type["units_expected"] / by_type["n_zip3"]

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
print("\n=== Y-B: demand by market type (Expected scenario) ===")
print(by_type)
print("\n=== Y-B-# (units) by scenario ===")
print(units_bt.round(0))
print("\n=== Y-B-$ (dollars) by scenario ===")
print(dollars_bt.round(0))

type_colors = {"Primary": "#1B4F72", "Secondary": "#5DADE2", "Tertiary": "#D5D8DC"}
pie_colors = [type_colors[t] for t in TYPES]

# ---------- Figure 4: pie chart (Expected scenario) ----------
fig, ax = plt.subplots(figsize=(7.5, 6))
vals = by_type["units_expected"].values
labels = [f"{t}\n{u:,.0f} units\n${d / 1e9:,.2f}B"
          for t, u, d in zip(TYPES, by_type["units_expected"], by_type["dollars_expected"])]
wedges, texts, autotexts = ax.pie(
    vals, labels=labels, colors=pie_colors, startangle=90, counterclock=False,
    autopct="%1.1f%%", pctdistance=0.72, textprops={"fontsize": 10},
    wedgeprops={"edgecolor": "white", "linewidth": 2})
for a, t in zip(autotexts, TYPES):
    a.set_color("white" if t == "Primary" else "black")
    a.set_fontweight("bold")
    a.set_fontsize(12)
ax.set_title("Y-B-# / Y-B-$  Annual market demand by market type\n"
             f"(Expected scenario: {df.loc['Expected', 'Units (#)']:,.0f} units, "
             f"${df.loc['Expected', 'Dollars ($)'] / 1e9:,.2f}B)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig4_YB_pie_expected.png"), dpi=200)

# ---------- Figure 5: stacked bars by scenario (units & dollars) ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, data, scale, unit_lbl, fmt, title in [
    (axes[0], units_bt, 1e6, "Million units", lambda v: f"{v:.2f}M", "Y-B-#  Units"),
    (axes[1], dollars_bt, 1e9, "Billion USD", lambda v: f"${v:.2f}B", "Y-B-$  Dollars"),
]:
    bottom = np.zeros(len(scen))
    for t in TYPES:
        v = data.loc[t, scen].values / scale
        bars = ax.bar(scen, v, bottom=bottom, color=type_colors[t],
                      label=t, edgecolor="white")
        for b, val, bt in zip(bars, v, bottom):
            ax.text(b.get_x() + b.get_width() / 2, bt + val / 2,
                    f"{fmt(val)}\n({by_type.loc[t, 'share']:.1%})",
                    ha="center", va="center", fontsize=9,
                    color="white" if t == "Primary" else "black")
        bottom += v
    for i, tot in enumerate(bottom):
        ax.text(i, tot * 1.01, fmt(tot), ha="center", va="bottom",
                fontweight="bold")
    ax.set_title(title)
    ax.set_ylabel(unit_lbl)
    ax.set_ylim(0, bottom.max() * 1.10)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(title="Market type", loc="lower right")
fig.suptitle("Annual market demand by market type and growth scenario",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig5_YB_stacked_scenarios.png"), dpi=200)

# ---------- Figure 6: share of ZIP3s vs share of demand (concentration) ----------
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(3)
w = 0.38
b1 = ax.bar(x - w / 2, by_type["zip3_pct"] * 100, w, color="#AEB6BF",
            label="% of ZIP3 areas")
b2 = ax.bar(x + w / 2, by_type["share"] * 100, w, color="#1B4F72",
            label="% of market demand")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
            f"{b.get_height():.1f}%", ha="center", fontweight="bold")
ax.set_xticks(x, TYPES)
ax.set_ylabel("%")
ax.set_title("Geographic footprint vs. demand share by market type",
             fontweight="bold")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig6_YB_zip_vs_demand.png"), dpi=200)

# ---------- Save tables ----------
by_type.to_csv(os.path.join(DATA_DIR, "task1_YB_by_market_type.csv"))
units_bt.to_csv(os.path.join(DATA_DIR, "task1_YB_units_by_scenario.csv"))
dollars_bt.to_csv(os.path.join(DATA_DIR, "task1_YB_dollars_by_scenario.csv"))

# ======================================================================
# Part 3: Y-C-#  (annual demand by state)
# ======================================================================
# ---------- 1. Aggregate PMF to state ----------
z_state = zmkt[["zip3", "State"]].drop_duplicates("zip3")
zs = z.merge(z_state, on="zip3", how="left")
zs["State"] = zs["State"].str.strip().str.upper()
print("\nZIP3 with missing State:", zs["State"].isna().sum())

by_state = (zs.groupby("State")
              .agg(n_zip3=("zip3", "count"), share=("pmf", "sum"))
              .sort_values("share", ascending=False))
by_state["units_expected"] = df.loc["Expected", "Units (#)"] * by_state["share"]
by_state["dollars_expected"] = by_state["units_expected"] * PRICE
by_state["units_conservative"] = df.loc["Conservative", "Units (#)"] * by_state["share"]
by_state["units_optimistic"] = df.loc["Optimistic", "Units (#)"] * by_state["share"]
by_state["cum_share"] = by_state["share"].cumsum()
by_state["rank"] = range(1, len(by_state) + 1)

print("\n=== Y-C-#: top 15 states (Expected) ===")
print(by_state.head(15)[["share", "cum_share", "units_expected", "dollars_expected"]])
for k in (5, 10, 15, 20):
    k = min(k, len(by_state))
    print(f"Top {k:>2} states = {by_state['cum_share'].iloc[k - 1]:.1%} of demand")
n80 = int((by_state["cum_share"] < 0.8).sum() + 1)
print(f"States needed to cover 80% of demand: {n80} of {len(by_state)}")

# Which market types make up each top state? (useful for insight)
mix = (zs.pivot_table(index="State", columns="market_type", values="pmf",
                      aggfunc="sum", fill_value=0)
         .reindex(columns=TYPES, fill_value=0))
mix = mix.div(mix.sum(axis=1), axis=0)
by_state = by_state.join(mix.add_suffix("_pct"))

# ---------- 2. Figure 7: sorted bar chart (top N) + cumulative % line ----------
TOP_N = 15
top = by_state.head(TOP_N)
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(top.index, top["units_expected"] / 1e3, color="#2E86C1")
for b, (_, r) in zip(bars, top.iterrows()):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
            f"{r['units_expected'] / 1e3:,.0f}K\n({r['share']:.1%})",
            ha="center", va="bottom", fontsize=8.5)
ax.set_ylabel("Annual demand (thousand units)")
ax.set_ylim(0, top["units_expected"].max() / 1e3 * 1.18)
ax.spines[["top"]].set_visible(False)

ax2 = ax.twinx()
ax2.plot(top.index, top["cum_share"] * 100, color="#C0392B", marker="o", lw=2)
ax2.axhline(80, color="#C0392B", ls=":", lw=1)
ax2.set_ylabel("Cumulative % of US demand", color="#C0392B")
ax2.set_ylim(0, 105)
ax2.spines[["top"]].set_visible(False)
ax.set_title(f"Y-C-#  Top {TOP_N} states by annual market demand "
             f"(Expected: {df.loc['Expected', 'Units (#)']:,.0f} units)\n"
             f"Top {TOP_N} states = {top['cum_share'].iloc[-1]:.0%} of US demand",
             fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig7_YC_top_states_bar.png"), dpi=200)

# ---------- 3. Figure 8: state choropleth (plotly) ----------
try:
    import plotly.express as px
    map_df = by_state.reset_index()
    map_df["units_K"] = map_df["units_expected"] / 1e3
    fig_map = px.choropleth(
        map_df, locations="State", locationmode="USA-states", scope="usa",
        color="units_expected", color_continuous_scale="Blues",
        hover_name="State",
        hover_data={"State": False, "units_expected": ":,.0f",
                    "dollars_expected": ":$,.0f", "share": ":.2%"},
        labels={"units_expected": "Units / year", "dollars_expected": "$ / year",
                "share": "Demand share"},
        title="Y-C-#  Annual market demand by state (Expected scenario, units)")
    fig_map.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    fig_map.write_html(os.path.join(PLOT_DIR, "fig8_YC_choropleth.html"))
    try:                                           # PNG needs: pip install kaleido
        fig_map.write_image(os.path.join(PLOT_DIR, "fig8_YC_choropleth.png"),
                            width=1100, height=650, scale=2)
    except Exception as e:
        print("PNG export skipped (pip install kaleido to enable):", e)
except ImportError:
    print("plotly not installed -> pip install plotly  (choropleth skipped)")

by_state.to_csv(os.path.join(DATA_DIR, "task1_YC_by_state.csv"))

# ======================================================================
# Part 4: Y-D-#  (annual demand by 3-digit ZIP)
# ======================================================================
# ---------- 1. ZIP3-level demand table ----------
zd = zs.copy()                                         # zip3, market_type, pmf, State
zd["units_expected"] = df.loc["Expected", "Units (#)"] * zd["pmf"]
zd["dollars_expected"] = zd["units_expected"] * PRICE
zd = zd.sort_values("pmf", ascending=False).reset_index(drop=True)
zd["rank"] = zd.index + 1
zd["cum_share"] = zd["pmf"].cumsum()
zd["cum_zip_pct"] = zd["rank"] / len(zd)
zd["label"] = zd["zip3"] + " (" + zd["State"].fillna("?") + ")"

print("\n=== Y-D-#: top 15 ZIP3 (Expected) ===")
print(zd.head(15)[["zip3", "State", "market_type", "pmf", "cum_share", "units_expected"]])
print(f"Total ZIP3: {len(zd)} | ZIP3 with positive demand: {(zd['pmf'] > 0).sum()}")
pareto_pts = {}
for q in (0.5, 0.8, 0.9):
    n = int((zd["cum_share"] < q).sum() + 1)
    pareto_pts[q] = n
    print(f"{n:>4} ZIP3 ({n / len(zd):.1%} of ZIP3s) cover {q:.0%} of demand")

# ---------- 2. ZIP3 coordinates (needed for the bubble map) ----------
# Option A: your own file with ZIP3 (or ZIP5) + latitude + longitude columns.
# Option B (fallback): pgeocode downloads a US postal-code table -> pip install pgeocode
COORD_FILE = os.path.join(DATA_DIR, "zip3_coordinates.csv")


def load_zip3_coords():
    if os.path.exists(COORD_FILE):
        c = pd.read_csv(COORD_FILE, dtype=str, sep=None, engine="python",
                        encoding="utf-8-sig")
        c.columns = c.columns.str.replace("\ufeff", "", regex=False).str.strip()
        zc, la, lo = (find_col(c, ["zip"]), find_col(c, ["lat"]),
                      find_col(c, ["lon", "lng"]))
        out = pd.DataFrame({"zip3": c[zc].str.strip().str.zfill(5).str[:3]
                            if c[zc].str.strip().str.len().max() >= 5
                            else c[zc].str.strip().str.zfill(3),
                            "lat": pd.to_numeric(c[la]), "lon": pd.to_numeric(c[lo])})
        print("Coordinates loaded from", COORD_FILE)
    else:
        import pgeocode
        d = pgeocode.Nominatim("us")._data[["postal_code", "latitude", "longitude"]]
        d = d.dropna()
        out = pd.DataFrame({"zip3": d["postal_code"].astype(str).str.zfill(5).str[:3],
                            "lat": d["latitude"].astype(float),
                            "lon": d["longitude"].astype(float)})
        print("Coordinates built from pgeocode (ZIP5 centroids averaged to ZIP3)")
    return out.groupby("zip3", as_index=False).mean()


try:
    coords = load_zip3_coords()
    zg = zd.merge(coords, on="zip3", how="left")
    print("ZIP3 without coordinates:", zg["lat"].isna().sum())
    zg = zg.dropna(subset=["lat", "lon"])
    HAVE_COORDS = True
except Exception as e:
    HAVE_COORDS = False
    print("Bubble map skipped (no coordinates):", e)

# ---------- 3. Figure 9: ZIP3 bubble map ----------
if HAVE_COORDS:
    fig, ax = plt.subplots(figsize=(13, 7.5))
    for t in TYPES:
        g = zg[(zg["market_type"] == t) & (zg["pmf"] > 0)]
        ax.scatter(g["lon"], g["lat"], s=g["units_expected"] / 25,
                   color=type_colors[t] if t != "Tertiary" else "#F5B041",
                   alpha=0.55, edgecolor="white", linewidth=0.4, label=t)
    # label the 10 biggest ZIP3
    for _, r in zg.head(10).iterrows():
        ax.annotate(r["zip3"], (r["lon"], r["lat"]), fontsize=8, fontweight="bold",
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(-126, -66)
    ax.set_ylim(24, 50)                                   # continental US window
    ax.set_aspect(1.25)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    lg = ax.legend(title="Market type", loc="lower left", markerscale=0.5)
    for h in lg.legend_handles:
        h.set_sizes([80])
    ax.set_title("Y-D-#  Annual demand by 3-digit ZIP (Expected scenario)\n"
                 "bubble area ∝ units per year; AK/HI not shown", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig9_YD_bubble_map.png"), dpi=200)

    # interactive version (hover for details)
    try:
        import plotly.express as px
        m = zg[zg["pmf"] > 0].copy()
        fig_b = px.scatter_geo(
            m, lat="lat", lon="lon", size="units_expected", color="market_type",
            category_orders={"market_type": TYPES}, scope="usa",
            hover_name="label",
            hover_data={"lat": False, "lon": False, "units_expected": ":,.0f",
                        "pmf": ":.3%"},
            size_max=30, opacity=0.6,
            title="Y-D-#  Annual demand by ZIP3 (Expected scenario, units)")
        fig_b.write_html(os.path.join(PLOT_DIR, "fig9_YD_bubble_map.html"))
    except Exception as e:
        print("Interactive map skipped:", e)

# ---------- 4. Figure 10: Pareto curve ----------
fig, ax = plt.subplots(figsize=(9, 6))
x = zd["cum_zip_pct"].values * 100
y = zd["cum_share"].values * 100
ax.plot(x, y, color="#1B4F72", lw=2.5, label="Cumulative demand")
ax.fill_between(x, y, alpha=0.12, color="#2E86C1")
ax.plot([0, 100], [0, 100], color="gray", ls="--", lw=1, label="Uniform demand")
for q, n in pareto_pts.items():
    xp = n / len(zd) * 100
    ax.plot([xp, xp], [0, q * 100], color="#C0392B", ls=":", lw=1)
    ax.plot([0, xp], [q * 100, q * 100], color="#C0392B", ls=":", lw=1)
    ax.scatter([xp], [q * 100], color="#C0392B", zorder=5)
    ax.annotate(f"{n} ZIP3s ({xp:.0f}%) → {q:.0%} of demand", (xp, q * 100),
                xytext=(10, -14), textcoords="offset points", fontsize=9,
                color="#C0392B", fontweight="bold")
ax.set_xlabel("% of ZIP3 areas (ranked by demand, largest first)")
ax.set_ylabel("Cumulative % of US demand")
ax.set_xlim(0, 100)
ax.set_ylim(0, 102)
ax.set_title(f"Y-D-#  Pareto curve of demand concentration across {len(zd)} ZIP3s",
             fontweight="bold")
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig10_YD_pareto.png"), dpi=200)

# ---------- 5. Figure 11: stacked bars of the top ZIP3s ----------
TOP_Z = 10
topz = zd.head(TOP_Z)
cmap = plt.get_cmap("tab10")
fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.5, 1]})

# (a) top ZIP3 stacked, per market-growth scenario (units)
ax = axes[0]
bottom = np.zeros(len(scen))
for i, (_, r) in enumerate(topz.iterrows()):
    v = np.array([df.loc[s, "Units (#)"] * r["pmf"] for s in scen]) / 1e3
    ax.bar(scen, v, bottom=bottom, color=cmap(i), edgecolor="white",
           label=f"{r['label']} - {r['market_type']}")
    bottom += v
for i, tot in enumerate(bottom):
    ax.text(i, tot + 2, f"{tot:,.0f}K", ha="center", fontweight="bold")
ax.set_ylabel("Annual demand (thousand units)")
ax.set_ylim(0, bottom.max() * 1.1)
ax.set_title(f"(a) Top {TOP_Z} ZIP3s stacked by scenario\n"
             f"(= {topz['pmf'].sum():.1%} of US demand)")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2)
ax.spines[["top", "right"]].set_visible(False)

# (b) concentration tiers (Expected): top 10 / 11-50 / 51-100 / rest
ax = axes[1]
tiers = [("Top 10", 0, 10), ("11-50", 10, 50), ("51-100", 50, 100),
         ("101+", 100, len(zd))]
tier_colors = ["#1B4F72", "#2E86C1", "#85C1E9", "#D5D8DC"]
bottom = 0
for (name, a, b), col in zip(tiers, tier_colors):
    part = zd.iloc[a:b]["units_expected"].sum() / 1e3
    ax.bar(["Expected"], [part], bottom=bottom, color=col, edgecolor="white",
           width=0.5)
    ax.text(0, bottom + part / 2,
            f"ZIP3 {name}\n{part:,.0f}K ({part * 1e3 / df.loc['Expected', 'Units (#)']:.1%})",
            ha="center", va="center", fontsize=9,
            color="white" if col in tier_colors[:2] else "black")
    bottom += part
ax.set_ylabel("Annual demand (thousand units)")
ax.set_title("(b) Demand by ZIP3 rank tier (Expected)")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Y-D-#  Where the demand sits: top ZIP3s", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig11_YD_top_zip3_stacked.png"), dpi=200)

zd.to_csv(os.path.join(DATA_DIR, "task1_YD_by_zip3.csv"), index=False)

# ======================================================================
# Part 5: Daily view (D-A-V/W, D-B-#, M-B-#, D-C-W, D-D-#)
# Uniform daily demand = annual / 365 (no seasonality -> baseline only)
# ======================================================================
SHARE_T = share_next["Median (20%)"]     # Tsukumo's own share (median scenario)


def flows(units_day):
    """Daily flows in the four measures, from units/day (scalar or Series)."""
    out = pd.DataFrame({"units_day": units_day})
    out["dollars_day"] = out["units_day"] * PRICE
    out["ft3_day"] = out["units_day"] * VOLUME
    out["lbs_day"] = out["units_day"] * WEIGHT
    return out


# ---------- 1. D-A-V / D-A-W : national daily flows ----------
da = flows(pd.Series({s: df.loc[s, "Units / day"] for s in scen}))
da["tsukumo_units_day"] = da["units_day"] * SHARE_T
print("\n=== D-A: national daily demand (Tsukumo + competitors) ===")
print(da.T)
da.to_csv(os.path.join(DATA_DIR, "task1_DA_daily.csv"))

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, col, scale, lbl, unit in [
    (axes[0], "ft3_day", 1e3, "D-A-V  Daily volume", "K ft³"),
    (axes[1], "lbs_day", 1e3, "D-A-W  Daily weight", "K lbs"),
]:
    bars = ax.bar(scen, da[col] / scale, color=colors)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                f"{b.get_height():,.1f}{unit}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_title(lbl)
    ax.set_ylabel(unit.replace("K ", "Thousand "))
    ax.set_ylim(0, da[col].max() / scale * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("National average daily flows (uniform, no seasonality)",
             fontsize=12.5, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig12_DA_volume_weight.png"), dpi=200)

# ---------- 2. D-B-# and M-B-# : by market type ----------
daily_bt = units_bt / DAYS                # rows = type, cols = scenario
monthly_bt = units_bt / MONTHS
print("\n=== D-B-# (units/day) ===")
print(daily_bt.round(0))
print("\n=== M-B-# (units/month) ===")
print(monthly_bt.round(0))

exp_bt = flows(daily_bt["Expected"])
exp_bt.insert(0, "share", by_type["share"])
exp_bt["units_month"] = monthly_bt["Expected"]
exp_bt["tsukumo_units_day"] = exp_bt["units_day"] * SHARE_T
exp_bt.loc["Total"] = exp_bt.sum()
exp_bt.loc["Total", "share"] = by_type["share"].sum()
print("\n=== Market type daily flows (Expected) ===")
print(exp_bt)
exp_bt.to_csv(os.path.join(DATA_DIR, "task1_DB_daily_by_type.csv"))

# Figure 13: grouped bars of daily units by market type
fig, ax = plt.subplots(figsize=(10, 5.5))
w = 0.26
for i, s in enumerate(scen):
    bars = ax.bar(np.arange(3) + (i - 1) * w, daily_bt[s].values, w,
                  color=colors[i], label=s)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 15,
                f"{b.get_height():,.0f}", ha="center", fontsize=9)
ax.set_xticks(range(3), TYPES)
ax.set_ylabel("Units per day")
ax.set_ylim(0, daily_bt.values.max() * 1.15)
ax.set_title("D-B-#  Average daily demand by market type", fontweight="bold")
ax.legend(title="Market growth")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig13_DB_daily_units_by_type.png"), dpi=200)

# Figure 14: summary table (Expected) - daily and monthly
cell = [[f"{r['share']:.1%}", f"{r['units_day']:,.0f}", f"${r['dollars_day'] / 1e6:,.1f}M",
         f"{r['ft3_day']:,.0f}", f"{r['lbs_day']:,.0f}",
         f"{r['units_month']:,.0f}", f"{r['tsukumo_units_day']:,.0f}"]
        for _, r in exp_bt.iterrows()]
cols = ["Demand\nshare", "Units\n/ day", "$ / day", "ft³ / day", "lbs / day",
        "Units\n/ month", "Tsukumo\nunits / day"]
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.axis("off")
t = ax.table(cellText=cell, rowLabels=list(exp_bt.index), colLabels=cols,
             loc="center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(10)
t.scale(1, 2.0)
for (r, c), cellobj in t.get_celld().items():
    if r == 0:
        cellobj.set_facecolor("#1B4F72")
        cellobj.set_text_props(color="white", fontweight="bold")
    if r == len(exp_bt):
        cellobj.set_text_props(fontweight="bold")
ax.set_title("D-B / M-B  Daily & monthly demand by market type (Expected scenario, "
             "uniform)", fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig14_DB_MB_table.png"), dpi=200)

# ---------- 3. D-C-W (top states) and D-D-# (top ZIP3) ----------
TOP_S, TOP_ZIP = 10, 15
ds_state = flows(by_state["units_expected"] / DAYS)
ds_state.index = by_state.index
ds_state["share"] = by_state["share"]
ds_state["tsukumo_units_day"] = ds_state["units_day"] * SHARE_T
ds_top = ds_state.head(TOP_S)
print(f"\n=== D-C-W: top {TOP_S} states, daily flows (Expected) ===")
print(ds_top[["share", "units_day", "lbs_day", "ft3_day", "tsukumo_units_day"]])
ds_state.to_csv(os.path.join(DATA_DIR, "task1_DC_daily_by_state.csv"))

zd_day = zd.head(TOP_ZIP).copy()
zd_day["units_day"] = zd_day["units_expected"] / DAYS
zd_day["tsukumo_units_day"] = zd_day["units_day"] * SHARE_T
print(f"\n=== D-D-#: top {TOP_ZIP} ZIP3, daily units (Expected) ===")
print(zd_day[["zip3", "State", "market_type", "units_day", "tsukumo_units_day"]])
zd_day.to_csv(os.path.join(DATA_DIR, "task1_DD_daily_top_zip3.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
ax = axes[0]
y = np.arange(len(ds_top))
ax.barh(y, ds_top["lbs_day"] / 1e3, color="#2E86C1")
for yi, (st, r) in zip(y, ds_top.iterrows()):
    ax.text(r["lbs_day"] / 1e3 + 0.3, yi,
            f"{r['lbs_day'] / 1e3:,.1f}K lbs | {r['units_day']:,.0f} units | "
            f"{r['ft3_day']:,.0f} ft³",
            va="center", fontsize=8.5)
ax.set_yticks(y, ds_top.index)
ax.invert_yaxis()
ax.set_xlim(0, ds_top["lbs_day"].max() / 1e3 * 1.6)
ax.set_xlabel("Daily weight (thousand lbs)")
ax.set_title(f"D-C-W  Top {TOP_S} states: average daily flow")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
y = np.arange(len(zd_day))
zcol = {t: (type_colors[t] if t != "Tertiary" else "#F5B041") for t in TYPES}
ax.barh(y, zd_day["units_day"], color=[zcol[t] for t in zd_day["market_type"]])
for yi, (_, r) in zip(y, zd_day.iterrows()):
    ax.text(r["units_day"] + 0.5, yi, f"{r['units_day']:,.0f} units/day",
            va="center", fontsize=8.5)
ax.set_yticks(y, zd_day["label"])
ax.invert_yaxis()
ax.set_xlim(0, zd_day["units_day"].max() * 1.3)
ax.set_xlabel("Units per day")
ax.set_title(f"D-D-#  Top {TOP_ZIP} ZIP3s: average daily units")
ax.spines[["top", "right"]].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=zcol[t], label=t) for t in TYPES], title="Market type",
          loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
fig.suptitle("Local daily handling scale (Tsukumo + competitors, uniform demand)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig15_DC_DD_top_states_zip3.png"), dpi=200)
