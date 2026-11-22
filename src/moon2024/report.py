"""The replication report: our results next to the paper's, as one HTML page.

  python -m moon2024.report [--outputs outputs] [--out docs/report.html]

Every number in the tables is computed here from our saved outputs and the
published values in paper.py; nothing is typed by hand. The page loads its
images from docs/figures (ours) and docs/paper_figures (the paper's, CC BY 4.0),
so it opens from the repository folder or from GitHub Pages.
"""

import argparse
import html
import json
from datetime import date
from pathlib import Path

from . import paper
from .compare import DATASET_NAMES, INPUT_NAMES, ensemble_scores
from .features import INPUT_SETS

METRICS = ("MAPE", "CVRMSE", "NMAE")
DEEP_NAMES = {"lstm": "LSTM", "bilstm": "Bi-LSTM", "gru": "GRU", "bigru": "Bi-GRU", "1d-cnn": "1D-CNN", "tcn": "TCN",
              "lstm-tcn": "LSTM-TCN", "bilstm-tcn": "Bi-LSTM-TCN", "gru-tcn": "GRU-TCN", "bigru-tcn": "Bi-GRU-TCN"}

# Figure number -> (our panels, how close, notes). Judged by eye, side by side; the images are below each note.
FIGURES = {
    2: (["fig02"], "Very close", "Taylor diagram, dormitory: 15 points (5 ensembles × 3 input sets)."),
    3: (["fig03"], "Very close",
        "Taylor diagram, AEP. The article's text says correlation \"about 0.6\"; its diagram, like ours, shows about "
        "0.3–0.5."),
    4: (["fig04a", "fig04b"], "Exact", "GBM, dormitory: feature importance and SHAP summary."),
    5: (["fig05a", "fig05b", "fig05c", "fig05d"], "Near-exact",
        "LightGBM and CatBoost in the same order; RF with one neighbouring swap; XGBoost with one swap and Temp two "
        "places higher. RF and CatBoost SHAP use a random 2,000 of the 20,472 training hours (exact SHAP on all was "
        "estimated above the 30-minute budget); their tails are visibly shorter than the paper's, probably because of "
        "the sample (not tested)."),
    6: (["fig06a", "fig06b"], "Close",
        "PDP of Holi × Cons_avg: top contour label 409.04 vs the paper's 412.24; labelled levels within 18. The paper's 6b "
        "lacks the extreme SHAP values its own Figure 4b shows for the same GBM, so 6b seems to come from a different "
        "run or subset of hours."),
    7: (["fig07a", "fig07b"], "Close", "PDP of THI × Cons_avg: labelled levels within 21; ours rises at THI above 72 "
        "and Cons_avg below 1,000, where the paper's shows only a small step, and labels one more top level. Same remark "
        "on 7b as on 6b."),
    8: (["fig08a", "fig08b"], "Exact", "CatBoost, AEP: feature importance and SHAP summary."),
    9: (["fig09a", "fig09b", "fig09c", "fig09d"], "Near-exact",
        "RF exact; XGBoost and LightGBM nearly; GBM somewhat different (GBM and XGBoost also keep differences of up to "
        "2.7 and 3.3 points in Tables 8–10)."),
    10: (["fig10a", "fig10b"], "Exact", "CatBoost, AEP: two-dimensional dependence plots."),
    11: (["fig11"], "Exact",
         "The article says this is GBM on the test set. It reproduces exactly only as CatBoost on the 2,311 training "
         "hours (we drew it four ways)."),
    12: (["fig12"], "RF, LightGBM scatter almost identical; others differ",
         "Ours are fitted and scored on all 3,289 hours (in-sample). For RF and LightGBM, whose scatters we reproduce, the "
         "paper's R² equals the squared correlation of these in-sample predictions: 0.964 (paper 0.964) and 0.831 "
         "(0.831). GBM, XGBoost and CatBoost match neither way (table below), so for them the definition can't be checked. The bars below show the usual R². The article doesn't give this experiment's settings, so ours come from the "
         "grid search. Held-out R² is in the table below."),
    13: (["fig13"], "Very close",
         "RF SHAP on the 27 raw columns: the top 13 and the bottom 5 in the paper's order. WeekStatus is coded "
         "weekend = 1 (the article's data section), which also gives the figure's colours."),
}

# Statements in the article that its own results don't support. Each quote is copied from the article text.
PAPER_ISSUES = [
    ("The described tuning doesn't produce the published tables.",
     "\"Hyperparameters were optimized through a five-fold cross-validation process … GridSearchCV\"",
     "Run as in the authors' grid code, the search gives AEP results up to 16 MAPE points (18 CVRMSE points) away from Tables 8–10 "
     "(both on weekdays; up to 11.8 points in the harmonic means of the tuning table). The fixed setting per model and "
     "dataset hard-coded in the authors' explainability notebook, reused for all three input sets, reproduces the tables "
     "exactly for four model-dataset pairs and most closely for the rest (up to 3.3 points on AEP)."),
    ("The Conclusions and the results section disagree on the best AEP model.",
     "\"the GBM model, honed with only external factors, outperformed its counterparts … within the AEP dataset\"",
     "The results section says \"the CatBoost model exhibited superior performance on the AEP dataset\", and Table 12 "
     "gives the lowest harmonic mean to CatBoost with all 15 inputs (48.23). GBM is best only among the external-input "
     "models (50.82), which may be what the Conclusions mean; read that way they are misleading rather than wrong."),
    ("The results text is ambiguous about the best input set.",
     "\"the statistical values attained their nadir when external factors were considered\"",
     "Read as external inputs alone, it is wrong: in Tables 5–10, all 15 inputs (which include the external ones) give "
     "the lowest value in all 45 dormitory cells and in 38 of 45 AEP cells; external inputs alone in only 7 AEP cells, "
     "and the next paragraph itself says the combination improved most. By harmonic mean, all 15 inputs are best for every model "
     "except XGBoost on AEP."),
    ("Figure 11 is described as the wrong model and data.",
     "\"Fig 11 offers a clear interpretation of the GBM model's predictions for the test set\"",
     "The figure reproduces exactly as CatBoost on the training hours."),
    ("Figure 12's R² values are not explained and appear to be in-sample.",
     "\"Fig 12 indicates that the RF model achieved the most commendable R2 value\"",
     "The article doesn't say how R² was computed. Refitted on all hours, our RF and LightGBM give almost the same "
     "scatter as the paper's, and the squared correlation of those in-sample predictions gives LightGBM 0.831 (paper "
     "0.831) and RF 0.964 (0.964); the usual R² is 0.706 and 0.930. GBM, XGBoost and CatBoost match neither way. Squared correlation ignores bias. On held-out "
     "weeks every model scores R² ≤ 0.05."),
    ("The text on Figure 3 overstates the correlation.",
     "\"Although a moderate correlation of about 0.6 was achieved\"",
     "The diagram shows about 0.3–0.5 (ours 0.28–0.51)."),
    ("The text on Figure 6(a) reverses the holiday effect.",
     "\"the plot shows that energy consumption peaks during holidays or weekends, highlighted by contours that rise to +412.24\"",
     "Both panels of Figure 6 (the paper's and ours) show days off lowering the prediction, and so does the data: the "
     "dormitory uses 1,291 kWh per hour on working days and 1,065 on days off. The +412 contour belongs to high Cons_avg, "
     "not to holidays. The text on Figure 6(b) reads the effect correctly."),
    ("The holiday indicator isn't built as described.",
     "\"we extracted holiday information from Time and Date … public holidays in both datasets\"",
     "The authors' published AEP feature table marks weekends only (e.g. Ascension Day, 5 May 2016, is 0); we follow "
     "the table that produced the results."),
    ("The Figure 13 text reverses WeekStatus.",
     "\"the week status (indicating a weekend (0) or weekday (1))\"",
     "Table 2 and the text on the inputs code weekday 0, weekend 1."),
    ("The Table 14 paragraph has garbled numbers.",
     "\"The respective values were 61 and 48.17 for CVRMSE, 87.31 and 92.46 for NMAE, and 48.66 and 55.78 for MAPE.\"",
     "The metric labels are shifted against Table 14 (48.17 is Bi-GRU's MAPE, 87.31 and 92.46 are CVRMSE), and \"61\" "
     "matches no value in the table (Bi-LSTM's MAPE is 37.61)."),
    ("Early stopping isn't mentioned.",
     "\"we ran 100 epochs to balance adequate learning and prevent overfitting\"",
     "The authors' notebook stops training early on a 10% validation split (patience 10). On AEP this matters: "
     "trained that way, our networks no longer produce exploding predictions (Table 14 above; predictions from -258 to 3,235 Wh instead of -4,113 to 6,077). Trained "
     "on the same 90% of windows for 100 epochs without stopping, the mean MAPE is 115.9 against 61.2 with stopping, so "
     "the gain comes from stopping (Beyond the paper)."),
]


def esc(text) -> str:
    return html.escape(str(text))


def diff_class(ours: float, theirs: float) -> str:
    d = abs(round(ours, 2) - theirs)
    return "exact" if d < 0.005 else "close" if d <= 0.1 + 1e-9 else "near" if d <= 1 + 1e-9 else "far"


def relative_class(ours: float, theirs: float) -> str:
    """For the deep networks, where exact reproduction isn't expected: the gap as a share of the paper's value."""
    r = abs(round(ours, 2) - theirs) / theirs
    return "exact" if r < 1e-9 else "close" if r <= 0.05 else "near" if r <= 0.15 else "far"


def cell(ours: float | None, theirs: float) -> str:
    if ours is None:
        return f'<td class="missing">–<span class="paper">{theirs:.2f}</span></td>'
    d = round(ours, 2) - theirs
    title = f"ours {ours:.2f}, paper {theirs:.2f}, difference {d + 0.0:+.2f}"
    return (f'<td class="{diff_class(ours, theirs)}" title="{title}">{ours:.2f}'
            f'<span class="paper">{theirs:.2f}</span></td>')


def ensemble_section(results: Path) -> tuple[str, dict]:
    counts = {}  # (dataset) -> [exact, total, max diff]
    parts = []
    for (dataset, inputs), (number, rows) in paper.ENSEMBLE_TABLES.items():
        header = "".join(f"<th>{m}<br><small>{g}</small></th>" for m in METRICS for g in "WHT")
        body = []
        for model, theirs in rows.items():
            ours = ensemble_scores(results, dataset, model, inputs)
            cells = []
            for column, t in zip(paper.COLUMNS, theirs):
                o = None if ours is None else ours[column]
                cells.append(cell(o, t))
                c = counts.setdefault(dataset, [0, 0, 0.0])
                if o is not None:
                    d = abs(round(o, 2) - t)
                    c[0] += d < 0.005
                    c[1] += 1
                    c[2] = max(c[2], d)
            body.append(f"<tr><th>{paper.MODEL_NAMES[model]}</th>{''.join(cells)}</tr>")
        parts.append(f'<h3>Table {number}: {DATASET_NAMES[dataset]}, {INPUT_NAMES[inputs]}</h3>'
                     f'<div class="scroll"><table class="cmp"><tr><th></th>{header}</tr>{"".join(body)}</table></div>')
    for dataset, (number, rows) in paper.HARMONIC_MEAN_TABLES.items():
        body = []
        for model, theirs in rows.items():
            cells = []
            for inputs, t in zip(INPUT_SETS, theirs):
                ours = ensemble_scores(results, dataset, model, inputs)
                cells.append(cell(None if ours is None else ours["HM_T"], t))
            body.append(f"<tr><th>{paper.MODEL_NAMES[model]}</th>{''.join(cells)}</tr>")
        parts.append(f'<h3>Table {number}: harmonic mean, {DATASET_NAMES[dataset]}</h3>'
                     f'<div class="scroll"><table class="cmp"><tr><th></th><th>External</th><th>Internal</th>'
                     f'<th>All 15</th></tr>{"".join(body)}</table></div>')
    return "".join(parts), counts


def tuning_section(fixed: Path, grid: Path) -> str:
    """Harmonic mean (all days) with the described grid search vs the authors' fixed settings vs the paper."""
    parts = []
    for dataset, (number, rows) in paper.HARMONIC_MEAN_TABLES.items():
        body = []
        for model, theirs in rows.items():
            cells = []
            for inputs, t in zip(INPUT_SETS, theirs):
                g, f = ensemble_scores(grid, dataset, model, inputs), ensemble_scores(fixed, dataset, model, inputs)
                gv = "–" if g is None else f"{g['HM_T']:.2f}"
                fv = "–" if f is None else f"{f['HM_T']:.2f}"
                gc = "" if g is None else diff_class(g["HM_T"], t)
                fc = "" if f is None else diff_class(f["HM_T"], t)
                cells.append(f'<td class="{gc}">{gv}</td><td class="{fc}">{fv}</td><td class="plain">{t:.2f}</td>')
            body.append(f"<tr><th>{paper.MODEL_NAMES[model]}</th>{''.join(cells)}</tr>")
        sets = "".join(f'<th colspan="3">{s}</th>' for s in ("External", "Internal", "All 15"))
        sub = "<th>grid</th><th>fixed</th><th>paper</th>" * 3
        parts.append(f"<h3>{DATASET_NAMES[dataset]} (compare Table {number})</h3>"
                     f'<div class="scroll"><table class="cmp tuning"><tr><th></th>{sets}</tr><tr><th></th>{sub}</tr>'
                     f'{"".join(body)}</table></div>')
    return "".join(parts)


def load_deep(folder: Path, dataset: str) -> dict:
    models = {}
    for path in sorted(folder.rglob(f"deep_{dataset}.json")):
        models.update(json.loads(path.read_text(encoding="utf-8"))["models"])
    return models


def deep_section(outputs: Path, results: Path) -> str:
    parts = []
    for dataset, (number, rows) in paper.DEEP_TABLES.items():
        runs = [("ours", load_deep(outputs / "deep", dataset), "hour24")]
        if dataset == "aep":
            runs.append(("ours, notebook early stopping", load_deep(outputs / "deep-es", dataset), "hour24"))
        head = "".join(f"<th>{m}</th>" for m in METRICS)
        body = []
        for model, theirs in rows.items():
            if model in DEEP_NAMES:
                name = DEEP_NAMES[model]
                lines = []
                for label, models, key in runs:
                    if model in models:
                        lines.append((label, [models[model][key][m] for m in METRICS]))
            else:
                name = f"{paper.MODEL_NAMES[model]} (best ensemble)"
                ours = ensemble_scores(results, dataset, model, "total")
                lines = [("ours", [ours[f"{m}_T"] for m in METRICS])] if ours is not None else []
            best = rows[[k for k in rows if k not in DEEP_NAMES][0]]
            cells = []
            for i, t in enumerate(theirs):
                beats = model in DEEP_NAMES and t < best[i]
                inner = "".join(f'<span class="run">{v[i]:.2f}<small>{esc(label)}</small></span>' for label, v in lines)
                colour = f" {relative_class(lines[0][1][i], t)}" if lines else ""  # main "ours" line against the paper
                cells.append(f'<td class="deep{colour}{" beats" if beats else ""}">{inner}'
                             f'<span class="run paperrun">{t:.2f}<small>paper</small></span></td>')
            body.append(f"<tr><th>{esc(name)}</th>{''.join(cells)}</tr>")
        parts.append(f"<h3>Table {number}: {DATASET_NAMES[dataset]}</h3>"
                     f'<div class="scroll"><table class="cmp deeptable"><tr><th></th>{head}</tr>{"".join(body)}'
                     f"</table></div>")
    return "".join(parts)


def figure_section(outputs: Path) -> str:
    raw = json.loads((outputs / "raw" / "raw_figures.json").read_text(encoding="utf-8"))["r2"]
    parts = []
    for number, (panels, verdict, note) in FIGURES.items():
        ours = "".join(f'<img loading="lazy" src="figures/{p}.png" alt="our {p}">' for p in panels)
        extra = ""
        if number == 12:
            rows = "".join(f"<tr><th>{paper.MODEL_NAMES[m]}</th><td>{v['paper']:.3f}</td><td>{v['in_sample_corr2']:.3f}</td>"
                           f"<td>{v['in_sample']:.3f}</td><td>{v['held_out']:.3f}</td></tr>" for m, v in raw.items())
            extra = ('<table class="cmp small"><tr><th></th><th>Paper R²</th><th>Ours, in-sample, squared correlation</th>'
                     "<th>Ours, in-sample, usual R²</th><th>Ours, held-out test weeks, usual R²</th></tr>"
                     f"{rows}</table>")
        parts.append(
            f'<section class="fig" id="fig{number}"><h3>Figure {number} <span class="verdict">{esc(verdict)}</span></h3>'
            f"<p>{esc(note)}</p>{extra}"
            f'<div class="pair"><figure><figcaption>Paper</figcaption>'
            f'<img loading="lazy" src="paper_figures/g{number:02d}.jpg" alt="paper figure {number}"></figure>'
            f'<figure><figcaption>Ours</figcaption><div class="panels n{len(panels)}">{ours}</div></figure></div>'
            f"</section>")
    return "".join(parts)


def markdown_to_html(text: str) -> str:
    """Just enough Markdown for the generated results files: headings, paragraphs, tables, bold."""
    import re

    def inline(t: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(t))

    out, table = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            if not set(line) <= set("|-: "):
                table.append([c.strip() for c in line.strip("|").split("|")])
            continue
        if table:
            head, *rows = table
            out.append('<div class="scroll"><table class="cmp"><tr>' + "".join(f"<th>{inline(c)}</th>" for c in head)
                       + "</tr>" + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows)
                       + "</table></div>")
            table = []
        if line.startswith("### "):
            out.append(f"<h4>{inline(line[4:])}</h4>")
        elif line.startswith("## "):
            out.append(f"<h3>{inline(line[3:])}</h3>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    return "".join(out)


def beyond_section(results: Path) -> str:
    """Our own analyses, generated by moon2024.significance and moon2024.stopping."""
    significance = (results / "significance.md").read_text(encoding="utf-8").split("\n", 2)[2]
    stopping = (results / "stopping_aep.md").read_text(encoding="utf-8")
    stopping = stopping[stopping.index("| Rule"):]  # the section has its own intro
    ours = load_deep(Path("outputs") / "deep", "aep")
    ours_mean = sum(m["hour24"]["MAPE"] for m in ours.values()) / len(ours)
    nets = [v[0] for k, v in paper.DEEP_TABLES["aep"][1].items() if k in DEEP_NAMES]
    paper_mean = sum(nets) / len(nets)
    return ("<h3>Are the differences significant?</h3>" + markdown_to_html(significance)
            + "<h3>When to stop training: the AEP deep networks, epoch by epoch</h3>"
            "<p>Each network was trained for 100 epochs on the first 90% of the training windows, with the test hours "
            "scored after every epoch (24th hour). Each row is a rule for choosing the epoch: a fixed epoch, or a rule that "
            "looks only at training and validation loss; the cells give the test MAPE there, and the last two columns "
            "are means over the ten networks. The last row picks by test MAPE and is shown only as a bound for MAPE. "
            f"Our run of the article's setup (all training windows, 100 epochs) has a mean MAPE of {ours_mean:.1f}; the "
            f"paper's Table 14 networks average {paper_mean:.1f}.</p>"
            + markdown_to_html(stopping)
            + '<figure class="wide"><img loading="lazy" src="figures/stopping_aep.png" alt="test MAPE, training and '
            'validation loss per epoch for the ten networks"><figcaption>MAPE axis capped at 200; 1D-CNN and TCN go higher in '
            'some epochs.</figcaption></figure>')


def issues_section() -> str:
    items = "".join(f"<li><strong>{esc(t)}</strong><blockquote>{esc(q)}</blockquote><p>{esc(e)}</p></li>"
                    for t, q, e in PAPER_ISSUES)
    return f'<ol class="issues">{items}</ol>'


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#deded8;--card:#ffffff;
--exact:#cfeccf;--close:#e4f2d0;--near:#fbeec2;--far:#f6cfc8;--accent:#2f5d8a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#e8e8e3;--muted:#9a9a93;
--line:#34342f;--card:#1f1f1d;--exact:#23452a;--close:#33452a;--near:#4a4224;--far:#532c27;--accent:#8ab4e0}}
:root[data-theme="dark"]{--bg:#161615;--fg:#e8e8e3;--muted:#9a9a93;--line:#34342f;--card:#1f1f1d;
--exact:#23452a;--close:#33452a;--near:#4a4224;--far:#532c27;--accent:#8ab4e0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.7rem;margin:.2em 0}h2{margin-top:2.4em;border-bottom:1px solid var(--line);padding-bottom:.3em}
h3{margin-top:1.8em}a{color:var(--accent)}
.cite{color:var(--muted);max-width:900px}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;z-index:2;
display:flex;gap:16px;flex-wrap:wrap;font-size:.9rem}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card b{display:block;font-size:.85rem;color:var(--muted);font-weight:600}
.scroll{overflow-x:auto}
table.cmp{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.9rem}
.cmp th,.cmp td{border:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}
.cmp th{background:var(--card);text-align:center}.cmp tr th:first-child{text-align:left}
.cmp td .paper{display:block;color:var(--muted);font-size:.78rem}
td.exact{background:var(--exact)}td.close{background:var(--close)}td.near{background:var(--near)}
td.far{background:var(--far)}td.plain{color:var(--muted)}
.legend span{display:inline-block;padding:2px 8px;margin-right:6px;border-radius:4px;font-size:.85rem}
.deep .run{display:block}.deep small{color:var(--muted);margin-left:6px;font-size:.72rem}
.deep .paperrun{color:var(--muted)}td.beats .paperrun{font-weight:700;color:var(--fg)}
.fig{border-top:1px solid var(--line);margin-top:28px}
.verdict{font-size:.8rem;font-weight:600;background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:2px 10px;margin-left:8px;vertical-align:middle}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}
.pair figure{margin:0;background:#fff;border:1px solid var(--line);border-radius:6px;padding:8px}
figcaption{font-weight:600;color:#333;margin-bottom:6px}
.pair img{width:100%;display:block}
.panels{display:grid;gap:6px}.panels.n2,.panels.n4{grid-template-columns:1fr 1fr}
figure.wide{margin:12px 0;background:#fff;border:1px solid var(--line);border-radius:6px;padding:8px}
figure.wide img{width:100%;display:block}
.issues li{margin-bottom:14px}blockquote{margin:4px 0;padding-left:10px;border-left:3px solid var(--line);
color:var(--muted);font-style:italic}
footer{margin-top:48px;color:var(--muted);font-size:.85rem}
@media (max-width:760px){.pair{grid-template-columns:1fr}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("docs/report.html"))
    args = parser.parse_args()
    results = args.outputs / "fixed"

    tables, counts = ensemble_section(results)
    cards = "".join(
        f'<div class="card"><b>Tree ensembles, {DATASET_NAMES[d]} (Tables {"5–7" if d == "urc" else "8–10"})</b>{c[0]} of {c[1]} '
        f"published values reproduced exactly; largest difference {c[2]:.2f} points</div>" for d, c in counts.items())
    cards += ('<div class="card"><b>Figures 2–13</b>4, 8, 10 match exactly, 11 exactly as CatBoost on the '
              "training hours; Taylor diagrams very close; 5, 9 nearly; 6, 7 closely; 13 very closely; 12 closely for RF and LightGBM</div>"
              '<div class="card"><b>Deep networks</b>Values not reproduced exactly; possible causes (untested): '
              "initialisation, GPU arithmetic, library versions. Dormitory: eight of ten networks within 1.8 points; our LSTM "
              "far better, our Bi-LSTM-TCN unstable. AEP (100 epochs): ours worse than the paper on MAPE for 9 "
              "of 10 networks, CVRMSE 8 of 10, NMAE 9 of 10; with the notebook's early stopping, MAPE worse for all 10 but "
              "CVRMSE lower than the paper's for 8 of 10</div>")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Moon 2024 Replication</title>
<style>{CSS}</style></head><body><main>
<h1>Replicating Moon et al. (2024)</h1>
<p class="cite">Moon, J., Maqsood, M., So, D., Baik, S. W., Rho, S., &amp; Nam, Y. (2024). Advancing ensemble learning
techniques for residential building electricity consumption forecasting: Insight from explainable artificial
intelligence. <i>PLOS ONE</i> 19(11): e0307654. <a href="https://doi.org/10.1371/journal.pone.0307654">doi</a>.
Both datasets: UCI Appliances Energy Prediction (AEP) and the University Residential Complex (dormitory) table from
the <a href="https://github.com/sodayeong/PLOS-ONE_Github">authors' repository</a>.</p>
<nav><a href="#summary">Summary</a><a href="#tables">Tables 5–12</a><a href="#tuning">Tuning</a>
<a href="#deep">Tables 13–14</a><a href="#figures">Figures</a><a href="#issues">Paper issues</a><a href="#beyond">Beyond the paper</a></nav>

<h2 id="summary">Summary</h2>
<div class="summary">{cards}</div>
<p>Data and inputs: Table 1 is reproduced exactly for both datasets, and our 15 inputs, built from the raw UCI data,
equal the authors' published AEP feature table on all 3,289 hours (automated tests).</p>

<h2 id="tables">Tree ensembles: Tables 5–12</h2>
<p>Each cell: ours, with the paper's value below. Colour by difference:</p>
<p class="legend"><span style="background:var(--exact)">exact</span><span style="background:var(--close)">≤ 0.1</span>
<span style="background:var(--near)">≤ 1</span><span style="background:var(--far)">&gt; 1</span></p>
<p>W = weekdays, H = holidays and weekends (AEP: weekends only), T = all days; all values in %. Trained with the authors' fixed settings
(next section).</p>
{tables}

<h2 id="tuning">Which settings produced the tables</h2>
<p>The article describes a 5-fold grid search per model; the authors' ensemble notebook runs it once per input set. Harmonic mean (all days), in %: <b>grid</b> =
the search as described, <b>fixed</b> = the single setting per model and dataset hard-coded in the authors' explainability notebook,
reused for every input set. The grid runs used scikit-learn 1.6.1 and the fixed runs 1.9.1, which alone moves single RF values by
up to 0.6 points; the large gaps come from the settings. Colour = difference from the paper, as above.</p>
{tuning_section(results, args.outputs / "ensembles")}

<h2 id="deep">Deep networks: Tables 13–14</h2>
<p>Ours are scored on the 24th forecast hour, as the article states. MAPE, CVRMSE, NMAE in %. A <b>bold</b> paper
value marks a network that beats the paper's best ensemble on that metric. Colour: how close our main value ("ours") is
to the paper's, relative to the paper's value: exact, within 5%, within 15%, more than 15%. (Tables 5–12 use absolute
points, because there exact reproduction is expected; for neural networks it isn't, and the two datasets' errors
differ in size.) The early-stopping line (AEP) is not coloured.</p>
{deep_section(args.outputs, results)}

<h2 id="figures">Figures 2–13</h2>
<p>The paper's figure on the left, ours on the right.</p>
{figure_section(args.outputs)}

<h2 id="issues">Statements in the article its results don't support</h2>
{issues_section()}

<h2 id="beyond">Beyond the paper: our own analyses</h2>
<p>Not part of the replication. The paper reports no significance tests, and doesn't mention early stopping or a validation split; these
sections look at both.</p>
{beyond_section(Path('docs/results'))}

<footer>Generated by <code>python -m moon2024.report</code> on {date.today().isoformat()} from
<a href="https://github.com/cafeina13/moon2024-replication">cafeina13/moon2024-replication</a>. Every number in the
tables is computed from saved outputs. The paper's figures and values are © 2024 Moon et al., published under
<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.</footer>
</main></body></html>
"""
    args.out.write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
