"""Ujian EDA Sains Data - analisis lanjutan IMSFD (dari metadata_IMSFD.csv sampai hasil akhir).
Pakai: python 02_ujian_eda_analysis.py --meta data/metadata_IMSFD.csv --out results
Parameter yang bisa diubah: --blank-thr --tiny-thr --cap --test-frac --seed --boot
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--meta", default="data/metadata_IMSFD.csv"); ap.add_argument("--out", default="results")
ap.add_argument("--blank-thr", type=float, default=5); ap.add_argument("--tiny-thr", type=int, default=64)
ap.add_argument("--cap", type=int, default=300); ap.add_argument("--test-frac", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--boot", type=int, default=400)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True); rng = np.random.default_rng(A.seed)
R = {}

def load(p):
    d = pd.read_csv(p); d["min_side"] = np.minimum(d.width, d.height)
    d["log_sharp"] = np.log1p(d.sharpness); d["log_size"] = np.log1p(d.file_size_kb); d["log_width"] = np.log(d.width)
    d["det"] = d.face_detected.astype(int); return d

def dupkey(x):
    k = x[["identity", "width", "height"]].copy()
    for c in ["file_size_kb", "brightness_mean", "sharpness"]: k[c] = x[c].round(4)
    return k

def clean(d, blank_thr, tiny_thr):
    log = {"n_awal": len(d)}; x = d.copy()
    m = x.brightness_mean < blank_thr; log["blank"] = int(m.sum()); x = x[~m]
    m = x.min_side < tiny_thr; log["tiny"] = int(m.sum()); x = x[~m]
    m = dupkey(x).duplicated(keep="first"); log["duplikat"] = int(m.sum()); x = x[~m]
    log["n_akhir"] = len(x); return x.copy(), log

def skew(s): return float(stats.skew(s.dropna()))
def iqr_out(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return float(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).mean() * 100)
def cliff(a, b):
    u = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    return 2 * u.statistic / (len(a) * len(b)) - 1, u.pvalue
def boot_ids(df, fn, B):  # cluster bootstrap: resample identitas
    g = {i: x for i, x in df.groupby("identity")}; ids = list(g); out = []
    for _ in range(B):
        s = rng.choice(len(ids), len(ids), replace=True); out.append(fn(pd.concat([g[ids[i]] for i in s])))
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]
def icc1(df, col):
    g = df.groupby("identity")[col]; k = g.size(); n = k.sum(); a = len(k)
    msb = (k * (g.mean() - df[col].mean()) ** 2).sum() / (a - 1); msw = ((df[col] - g.transform("mean")) ** 2).sum() / (n - a)
    k0 = (n - (k ** 2).sum() / n) / (a - 1); return float((msb - msw) / (msb + (k0 - 1) * msw)), float(k0)

def tests(df, B):
    r = {"N": len(df)}
    a, b = df[df.det == 1].sharpness, df[df.det == 0].sharpness
    dl, p = cliff(a, b); r["H1_delta"], r["H1_p"] = float(dl), float(p)
    r["H1_ci"] = boot_ids(df, lambda x: cliff(x[x.det == 1].sharpness, x[x.det == 0].sharpness)[0], B)
    r["H1_med_det"], r["H1_med_nodet"] = float(a.median()), float(b.median()); r["det_rate"] = float(df.det.mean())
    ct = pd.crosstab(df.subset, df.det); chi2, p2, _, _ = stats.chi2_contingency(ct)
    r["H2_chi2"], r["H2_p"] = float(chi2), float(p2); r["H2_V"] = float(np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1))))
    r["rate"] = df.groupby("subset").det.mean().round(4).to_dict()
    idr = df.groupby(["subset", "identity"]).det.mean().reset_index(); grp = [x.det.values for _, x in idr.groupby("subset")]
    H, pk = stats.kruskal(*grp); r["H2_KW_H"], r["H2_KW_p"] = float(H), float(pk); r["H2_eps2"] = float((H - len(grp) + 1) / (len(idr) - len(grp)))
    f = lambda x: x[x.subset == "B1"].det.mean() - x[x.subset == "A2"].det.mean()
    r["H2_diff_B1_A2"] = float(f(df)); r["H2_diff_ci"] = boot_ids(df, f, B)
    t = df[df.det == 1]; rho, p3 = stats.spearmanr(t.width, t.face_area_ratio)
    r["H3_rho"], r["H3_p"], r["H3_n"] = float(rho), float(p3), len(t)
    r["H3_ci"] = boot_ids(t, lambda x: stats.spearmanr(x.width, x.face_area_ratio)[0], B)
    r["H3_by_subset"] = {s: float(stats.spearmanr(x.width, x.face_area_ratio)[0]) for s, x in t.groupby("subset")}
    return r

def fit_logit(X, y, groups):  # IRLS + SE robust terhadap klaster identitas
    X = np.column_stack([np.ones(len(X)), X]); b = np.zeros(X.shape[1])
    for _ in range(30):
        p = 1 / (1 + np.exp(-X @ b)); H = X.T @ (X * (p * (1 - p))[:, None]); b = b + np.linalg.solve(H, X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ b)); H = X.T @ (X * (p * (1 - p))[:, None]); bread = np.linalg.inv(H)
    S = pd.DataFrame(X * (y - p)[:, None]).groupby(np.asarray(groups)).sum().values; G = len(S)
    V = bread @ (S.T @ S) @ bread * G / (G - 1); return b, np.sqrt(np.diag(V))

def logit(df):
    z = pd.DataFrame({"z_sharp": (df.log_sharp - df.log_sharp.mean()) / df.log_sharp.std(), "z_width": (df.log_width - df.log_width.mean()) / df.log_width.std(),
                      "z_bright": (df.brightness_mean - df.brightness_mean.mean()) / df.brightness_mean.std()})
    sd = pd.get_dummies(df.subset)[["A1", "B1"]].astype(float); X = pd.concat([z, sd], axis=1); X.index = df.index; y = df.det.values
    b, se = fit_logit(X.values, y, df.identity.values); out = {}
    for i, nm in enumerate(X.columns, start=1):
        out[nm] = {"OR": float(np.exp(b[i])), "lo": float(np.exp(b[i] - 1.96 * se[i])), "hi": float(np.exp(b[i] + 1.96 * se[i])), "p": float(2 * (1 - stats.norm.cdf(abs(b[i] / se[i]))))}
    au = {}
    for nm, cols in [("baseline_subset", ["A1", "B1"]), ("model_kualitas", list(X.columns))]:
        s = []
        for tr, te in GroupKFold(5).split(X, y, df.identity.values):
            c = LogisticRegression(max_iter=500).fit(X[cols].iloc[tr], y[tr]); s.append(roc_auc_score(y[te], c.predict_proba(X[cols].iloc[te])[:, 1]))
        au[nm] = float(np.mean(s))
    return out, au

# ---------- DATA MENTAH ----------
raw = load(A.meta); R["raw"] = {"N": len(raw), "ids": int(raw.identity.nunique())}
nb = raw[raw.brightness_mean >= A.blank_thr].copy(); nb["g"] = dupkey(nb).astype(str).agg("|".join, axis=1)
gs = nb.groupby("g").agg(n=("split", "size"), ns=("split", "nunique")); dg = gs[gs.n > 1]
R["dup_raw"] = {"extra_nonblank": int((nb.duplicated("g")).sum()), "groups": int(len(dg)), "groups_cross_split": int((dg.ns > 1).sum()),
                "test_imgs_with_train_twin": int(nb[nb.g.isin(dg[dg.ns > 1].index) & (nb.split == "testing")].shape[0]), "extra_all": int(dupkey(raw).duplicated().sum())}
# ---------- B: PERSIAPAN DATA ----------
cl, log = clean(raw, A.blank_thr, A.tiny_thr); R["clean_log"] = log; R["clean_by_subset"] = cl.groupby("subset").size().to_dict()
rows = []
for v in ["width", "file_size_kb", "sharpness", "brightness_mean", "face_area_ratio", "log_width", "log_size", "log_sharp"]:
    rows.append({"var": v, "mean_raw": raw[v].mean(), "med_raw": raw[v].median(), "skew_raw": skew(raw[v]), "iqrout_raw": iqr_out(raw[v]),
                 "mean_cl": cl[v].mean(), "med_cl": cl[v].median(), "skew_cl": skew(cl[v]), "iqrout_cl": iqr_out(cl[v])})
R["before_after"] = rows; R["det_raw"], R["det_clean"] = float(raw.det.mean()), float(cl.det.mean())
R["ar_gt15"] = {"raw": int((raw.aspect_ratio > 1.5).sum()), "clean": int((cl.aspect_ratio > 1.5).sum())}
R["minside_med_subset"] = {"raw": raw.groupby("subset").min_side.median().to_dict(), "clean": cl.groupby("subset").min_side.median().to_dict()}
R["rate_raw"] = raw.groupby("subset").det.mean().round(4).to_dict()
cl["split_v2"] = ""
for s, x in cl.groupby("subset"):
    tr, te = next(GroupShuffleSplit(1, test_size=A.test_frac, random_state=A.seed).split(x, groups=x.identity))
    cl.loc[x.index[tr], "split_v2"] = "train"; cl.loc[x.index[te], "split_v2"] = "test"
tri, tei = set(cl[cl.split_v2 == "train"].identity), set(cl[cl.split_v2 == "test"].identity)
cl["in_balanced"] = False
for i, x in cl.groupby("identity"): cl.loc[x.sample(min(len(x), A.cap), random_state=A.seed).index, "in_balanced"] = True
cb, ca = cl.groupby("identity").size(), cl[cl.in_balanced].groupby("identity").size()
R["split_v2"] = {"train_ids": len(tri), "test_ids": len(tei), "overlap_ids": len(tri & tei), "train_n": int((cl.split_v2 == "train").sum()), "test_n": int((cl.split_v2 == "test").sum()),
                 "dup_left": int(dupkey(cl).duplicated().sum())}
R["balance"] = {"cv_before": float(cb.std() / cb.mean()), "cv_after": float(ca.std() / ca.mean()), "min": int(cb.min()), "max_before": int(cb.max()), "n_balanced": int(cl.in_balanced.sum()), "ids_lt100": int((cb < 100).sum())}
cl.drop(columns=["log_size"]).to_csv(os.path.join(A.out, "metadata_clean.csv"), index=False)
# ---------- C: HIPOTESIS ----------
R["tests_clean"] = tests(cl, A.boot)
a1, a0 = cl[cl.det == 1].log_sharp, cl[cl.det == 0].log_sharp
R["assump"] = {"skew_sharp": skew(cl.sharpness), "skew_logsharp": skew(cl.log_sharp),
  "shapiro_sharp_p": float(stats.shapiro(cl.sharpness.sample(5000, random_state=1)).pvalue), "shapiro_logsharp_p": float(stats.shapiro(cl.log_sharp.sample(5000, random_state=1)).pvalue),
  "levene_logsharp_p": float(stats.levene(a1, a0).pvalue), "cohen_d_logsharp": float((a1.mean() - a0.mean()) / np.sqrt((a1.var() + a0.var()) / 2))}
i1, k0 = icc1(cl, "det"); i2, _ = icc1(cl, "log_sharp"); R["icc"] = {"det": i1, "log_sharp": i2, "m": k0, "deff_det": 1 + (k0 - 1) * i1}
R["logit"], R["auc"] = logit(cl)
# ---------- D: SENSITIVITAS ----------
q = cl.copy()
for c in ["sharpness", "width", "brightness_mean"]:
    q1, q3 = q[c].quantile([.25, .75]); i = q3 - q1; q = q[(q[c] >= q1 - 1.5 * i) & (q[c] <= q3 + 1.5 * i)]
sc = {"S0 Data mentah": raw, "S1 Pembersihan utama": cl, "S2 S1 tanpa outlier IQR": q,
      "S3 Ambang sisi 32 px": clean(raw, A.blank_thr, 32)[0], "S4 Ambang sisi 128 px": clean(raw, A.blank_thr, 128)[0],
      "S5 Tanpa deduplikasi": raw[(raw.brightness_mean >= A.blank_thr) & (raw.min_side >= A.tiny_thr)]}
R["sens"] = {}
for nm, x in sc.items():
    r = tests(x, max(100, A.boot // 2)); r["logit_sharp_OR"] = logit(x)[0]["z_sharp"]["OR"]; R["sens"][nm] = r
json.dump(R, open(os.path.join(A.out, "results.json"), "w"), indent=1, default=float)
# ---------- FIGUR ----------
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
f, ax = plt.subplots(2, 3, figsize=(7.2, 3.8))
for j, (v, tv, t) in enumerate([("sharpness", "log_sharp", "Sharpness"), ("file_size_kb", "log_size", "Ukuran file (KB)"), ("width", "log_width", "Lebar (px)")]):
    ax[0, j].hist(raw[v], bins=40, color="#c0504d"); ax[0, j].set_title(f"{t}: mentah (skew {skew(raw[v]):.1f})", fontsize=7)
    ax[1, j].hist(cl[tv], bins=40, color="#1f77b4"); ax[1, j].set_title(f"log, bersih (skew {skew(cl[tv]):.2f})", fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(A.out, "fig1_sebelum_sesudah.png"), dpi=200); plt.close()
T = R["tests_clean"]; f, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
ax[0].boxplot([a0, a1], tick_labels=["Tidak\nterdeteksi", "Terdeteksi"], showfliers=False)
rt = cl.groupby("subset").det.mean() * 100; ax[1].bar(rt.index, rt.values, color="#1f77b4"); ax[1].set_ylim(70, 100); ax[1].set_ylabel("Deteksi wajah (%)"); ax[1].set_title(f"H2: Cramér V = {T['H2_V']:.2f}")
for i, v in enumerate(rt.values): ax[1].text(i, v + .5, f"{v:.1f}", ha="center")
t = cl[cl.det == 1]; ax[2].hexbin(np.log10(t.width), t.face_area_ratio, gridsize=35, cmap="Blues", mincnt=1); ax[2].set_xlabel("log10(lebar px)"); ax[2].set_ylabel("Face area ratio"); ax[2].set_title(f"H3: Spearman rho = {T['H3_rho']:.2f}")
plt.tight_layout(); plt.savefig(os.path.join(A.out, "fig2_hipotesis.png"), dpi=200); plt.close()
names = list(R["sens"]); y = np.arange(len(names))[::-1]; f, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
for a, (k, ck, ttl) in zip(ax, [("H1_delta", "H1_ci", "H1: delta Cliff (sharpness)"), ("H3_rho", "H3_ci", "H3: Spearman rho (lebar vs face ratio)")]):
    v = np.array([R["sens"][n][k] for n in names]); lo = np.array([R["sens"][n][ck][0] for n in names]); hi = np.array([R["sens"][n][ck][1] for n in names])
    a.errorbar(v, y, xerr=[v - lo, hi - v], fmt="o", color="#1f77b4", capsize=2); a.set_title(ttl); a.axvline(0, color="grey", lw=.5)
ax[0].set_yticks(y); ax[0].set_yticklabels([n.split(" ")[0] + " " + " ".join(n.split(" ")[1:3]) for n in names]); plt.tight_layout(); plt.savefig(os.path.join(A.out, "fig3_sensitivitas.png"), dpi=200); plt.close()
print("Selesai. Hasil di", A.out)
