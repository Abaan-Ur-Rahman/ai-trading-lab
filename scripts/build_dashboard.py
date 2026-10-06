"""Build a single-page HTML dashboard from the project's reports.

    python scripts/build_dashboard.py      # writes reports/dashboard.html, then open it in a browser

Read-only: it only reads reports, the forward-test log and data, and the daily
update log, and writes one HTML file. It changes no data, model or rule, so it
is safe to run at any time, including during the forward test.

Shows: the daily update's health, forward-test progress, the one-time test
check, the walk-forward backtest's equity curves, and the pre-registered
forward rules' equity curves with a metrics table. Any piece whose inputs
don't exist yet is shown as "not available yet" rather than failing.

Charts load Chart.js from cdnjs, so viewing the page needs an internet
connection; the data itself is embedded in the file.
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd

from backtest.engine import buy_and_hold, run_backtest
from forward.tracking import attach_outcomes
from forward_test import (
    HORIZON,
    MIN_BARS_FOR_VERDICT,
    PREDICTIONS_PATH,
    PRIMARY,
    RULES,
    RULES_REGISTERED_ON,
    THRESHOLD,
    evaluate_rule,
    history,
    load_csv,
)
from ml.dataset import CLASS_NAMES

REPORTS = PROJECT_ROOT / "reports"
OUTPUT = REPORTS / "dashboard.html"
UPDATE_LOG = PROJECT_ROOT / "logs" / "forward_update.log"
RULE_LABELS = {
    "A_current": "A: current rules",
    "B_confident": "B: confidence ≥ 0.55",
    "C_long_only": "C: long only",
    "D_longer_hold": "D: 24-bar hold",
}


def update_health() -> dict | None:
    """Last run time and exit code of the scheduled daily update, from its log."""
    if not UPDATE_LOG.exists():
        return None
    text = UPDATE_LOG.read_text(encoding="utf-8", errors="replace").replace("﻿", "")
    runs = re.findall(r"===== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) =====(.*?)(?====== |\Z)", text, re.S)
    if not runs:
        return None
    parsed = []
    for stamp, body in runs:
        code = re.search(r"exit code: (-?\d+)", body)
        parsed.append((pd.Timestamp(stamp), int(code.group(1)) if code else None))
    last_time, last_code = parsed[-1]
    successes = [t for t, c in parsed if c == 0]
    return {
        "last_run": f"{last_time:%Y-%m-%d %H:%M}",
        "last_ok": last_code == 0,
        "last_code": last_code,
        "runs": len(parsed),
        "last_success": f"{successes[-1]:%Y-%m-%d %H:%M}" if successes else None,
        "hours_since_success": (pd.Timestamp.now() - successes[-1]).total_seconds() / 3600 if successes else None,
    }


def final_test_check() -> dict | None:
    path = REPORTS / "final_test_check.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    means = {name: sum(c["macro_f1_by_seed"]) / len(c["macro_f1_by_seed"]) for name, c in record["configs"].items()}
    return {"decision": record["decision"], "gap": record["macro_f1_mean_gap"],
            "wins": record["seeds_beating_base"], "base": means.get("base"), "combined": means.get("plus_combined")}


def backtest_section() -> dict | None:
    summary_path, equity_path = REPORTS / "backtest_walk_forward.json", REPORTS / "backtest_equity.csv"
    if not (summary_path.exists() and equity_path.exists()):
        return None
    summary = json.loads(summary_path.read_text())
    equity = pd.read_csv(equity_path, index_col="timestamp", parse_dates=True)
    equity = equity.resample("6h").last().dropna(how="all")  # hourly -> 6-hourly keeps the page light
    return {
        "labels": [f"{t:%Y-%m-%d %H:%M}" for t in equity.index],
        "series": {column: [None if pd.isna(v) else round(float(v), 2) for v in equity[column]] for column in equity},
        "summary": summary["summary"],
        "period": summary["period"],
        "seeds": summary["seeds"],
    }


def forward_section() -> dict | None:
    log = load_csv(PREDICTIONS_PATH)
    if log is None or log.empty:
        return None
    primary = history(PRIMARY[0])
    scored = attach_outcomes(log, primary, horizon=HORIZON, threshold=THRESHOLD)
    done = scored.dropna(subset=["actual"])
    section = {"logged": len(scored), "with_outcomes": len(done), "needed": MIN_BARS_FOR_VERDICT,
               "first_bar": f"{scored.index.min():%Y-%m-%d %H:%M}", "last_bar": f"{scored.index.max():%Y-%m-%d %H:%M}",
               "registered_on": RULES_REGISTERED_ON, "curves": None, "rules": []}
    if len(done) < 2:
        return section

    prices = primary.loc[done.index.min():]
    signals = done[CLASS_NAMES]
    curves = {}
    for name, config in RULES.items():
        result = run_backtest(prices, signals, config)
        curves[RULE_LABELS[name]] = result.equity
        m = result.metrics
        weekday = evaluate_rule(prices, signals, config)["weekday_only"]
        section["rules"].append({"name": RULE_LABELS[name], "return": m["total_return"], "sharpe": m["sharpe_ratio"],
                                 "drawdown": m["max_drawdown"], "trades": m["n_trades"], "win_rate": m["win_rate"],
                                 "weekday_return": weekday["total_return"] if weekday else None})
    benchmark = buy_and_hold(prices, done.index.min())
    curves["Buy & hold gold"] = benchmark.equity
    bm = benchmark.metrics
    section["rules"].append({"name": "Buy & hold gold", "return": bm["total_return"], "sharpe": bm["sharpe_ratio"],
                             "drawdown": bm["max_drawdown"], "trades": None, "win_rate": None,
                             "weekday_return": None})
    frame = pd.DataFrame(curves).ffill()
    section["curves"] = {
        "labels": [f"{t:%Y-%m-%d %H:%M}" for t in frame.index],
        "series": {c: [None if pd.isna(v) else round(float(v), 2) for v in frame[c]] for c in frame},
    }
    return section


def pct(value, digits=1, sign=True) -> str:
    if value is None:
        return "–"
    return f"{value:+.{digits}%}" if sign else f"{value:.{digits}%}"


def tile(label: str, value: str, detail: str, status: str | None = None) -> str:
    badge = ""
    if status:
        icon, text = {"good": ("✓", "OK"), "warning": ("!", "Check"), "critical": ("✕", "Failing"),
                      "neutral": ("…", "Waiting")}[status]
        badge = f'<span class="badge {status}"><span aria-hidden="true">{icon}</span> {text}</span>'
    return (f'<div class="tile"><div class="tile-label">{html.escape(label)}{badge}</div>'
            f'<div class="tile-value">{value}</div><div class="tile-detail">{detail}</div></div>')


def build() -> str:
    health, check, backtest, forward = update_health(), final_test_check(), backtest_section(), forward_section()

    tiles = []
    if health:
        stale = health["hours_since_success"] is None or health["hours_since_success"] > 36
        status = "critical" if not health["last_ok"] else ("warning" if stale else "good")
        detail = f"exit code {health['last_code']} · {health['runs']} runs logged"
        if stale and health["last_ok"]:
            detail += " · no successful run in 36h"
        tiles.append(tile("Daily update", html.escape(health["last_run"]), detail, status))
    else:
        tiles.append(tile("Daily update", "No log yet", "logs/forward_update.log not found", "neutral"))

    if forward:
        share = forward["with_outcomes"] / forward["needed"]
        bar = (f'<div class="progress" role="progressbar" aria-valuenow="{forward["with_outcomes"]}" '
               f'aria-valuemax="{forward["needed"]}"><div style="width:{min(share, 1):.1%}"></div></div>')
        tiles.append(tile("Forward test", f'{forward["with_outcomes"]:,} / {forward["needed"]:,}',
                          f"bars with outcomes ({share:.0%}){bar}", "good" if share >= 1 else "neutral"))
    else:
        tiles.append(tile("Forward test", "Not started", "no predictions logged yet", "neutral"))

    if check:
        tiles.append(tile("One-time test check", "Confirmed" if check["decision"] == "confirmed" else "Not confirmed",
                          f"macro-F1 {check['combined']:.4f} vs {check['base']:.4f} · {check['wins']}/10 seeds"))
    if backtest:
        s = backtest["summary"]
        tiles.append(tile("Backtest, out-of-sample", pct(s["plus_combined"]["total_return"]["mean"], 0),
                          f"dollar features · buy &amp; hold {pct(s['buy_and_hold']['total_return']['mean'], 0)}"))

    data = {"backtest": backtest, "forward": forward}

    backtest_html = '<p class="empty">Backtest not available yet. Run scripts/backtest_walk_forward.py.</p>'
    if backtest:
        s = backtest["summary"]
        rows = "".join(
            f"<tr><th scope='row'>{label}</th><td>{pct(s['base'][key]['mean'])}</td>"
            f"<td>{pct(s['plus_combined'][key]['mean'])}</td><td>{pct(s['buy_and_hold'][key]['mean'])}</td></tr>"
            for key, label in [("total_return", "Total return"), ("annualized_return", "Annualised return"),
                               ("max_drawdown", "Max drawdown")]
        ) + (f"<tr><th scope='row'>Sharpe</th><td>{s['base']['sharpe_ratio']['mean']:+.2f}</td>"
             f"<td>{s['plus_combined']['sharpe_ratio']['mean']:+.2f}</td>"
             f"<td>{s['buy_and_hold']['sharpe_ratio']['mean']:+.2f}</td></tr>"
             f"<tr><th scope='row'>Trades</th><td>{s['base']['n_trades']['mean']:.0f}</td>"
             f"<td>{s['plus_combined']['n_trades']['mean']:.0f}</td><td>–</td></tr>")
        backtest_html = f"""
        <p class="caption">$10,000 start, monthly retraining on past data only, {backtest['period'][0][:10]} to
        {backtest['period'][1][:10]}. Lines show seed 0; the table shows the mean of {backtest['seeds']} seeds.
        0.05% cost per side.</p>
        <div class="chart"><canvas id="backtestChart" aria-label="Backtest equity curves"></canvas></div>
        <details><summary>Table view</summary><div class='table-wrap'><table><thead><tr><th></th><th>Base</th><th>+ EUR/USD + USD/JPY</th>
        <th>Buy &amp; hold</th></tr></thead><tbody>{rows}</tbody></table></div></details>"""

    forward_html = '<p class="empty">No forward predictions yet. The daily update will start logging them.</p>'
    if forward:
        early = ""
        if forward["with_outcomes"] < forward["needed"]:
            early = (f'<div class="notice"><span class="badge neutral"><span aria-hidden="true">…</span> Too early</span>'
                     f' Only {forward["with_outcomes"]:,} of the {forward["needed"]:,} bars needed have outcomes. '
                     "Early curves swing a lot; don't change anything because of them.</div>")
        table, chart = "", ""
        if forward["rules"]:
            rows = "".join(
                f"<tr><th scope='row'>{html.escape(r['name'])}</th><td>{pct(r['return'])}</td><td>{r['sharpe']:+.2f}</td>"
                f"<td>{pct(r['drawdown'], sign=False)}</td><td>{'–' if r['trades'] is None else r['trades']}</td>"
                f"<td>{'–' if r['win_rate'] is None else pct(r['win_rate'], sign=False)}</td>"
                f"<td>{pct(r['weekday_return'])}</td></tr>"
                for r in forward["rules"])
            table = (f"<div class='table-wrap'><table><thead><tr><th>Rule</th><th>Return</th><th>Sharpe</th><th>Max drawdown</th><th>Trades</th>"
                     f"<th>Win rate</th><th>Weekday-only return</th></tr></thead><tbody>{rows}</tbody></table></div>")
            chart = '<div class="chart"><canvas id="forwardChart" aria-label="Forward rule equity curves"></canvas></div>'
        forward_html = f"""
        <p class="caption">Production model's predictions on bars after {forward['first_bar']} UTC (latest
        {forward['last_bar']}). Rules registered {forward['registered_on']}; a rule passes only if, after
        {forward['needed']:,} bars, its return after costs and its Sharpe are positive on all bars and on weekday-only trades
        (weekend quotes can't be traded).</p>
        {early}{chart}{table}"""

    generated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return (TEMPLATE.replace("__TILES__", "".join(tiles)).replace("__BACKTEST__", backtest_html)
            .replace("__FORWARD__", forward_html).replace("__GENERATED__", generated)
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Trading Lab Dashboard</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
    --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --ref: #898781;
    --good: #0ca30c; --warning: #fab219; --critical: #d03b3b; --neutral: #898781; --good-text: #006300;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
      --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --ref: #898781; --good-text: #0ca30c;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --ref: #898781; --good-text: #0ca30c;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink);
         font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 32px 16px 48px; }
  header { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; flex-wrap: wrap; }
  h1 { font-size: 22px; margin: 0; } h2 { font-size: 17px; margin: 0 0 4px; }
  .generated { color: var(--muted); font-size: 13px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 24px 0; }
  .tile, section { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
  .tile-label { color: var(--ink-2); font-size: 13px; display: flex; justify-content: space-between; gap: 8px; }
  .tile-value { font-size: 26px; font-weight: 600; margin: 6px 0 2px; }
  .tile-detail { color: var(--ink-2); font-size: 13px; }
  .badge { font-size: 12px; font-weight: 600; white-space: nowrap; color: var(--ink); }
  .badge > span { display: inline-block; width: 16px; height: 16px; line-height: 16px; text-align: center;
                  border-radius: 50%; color: #fff; font-size: 11px; margin-right: 2px; }
  .badge.good > span { background: var(--good); } .badge.warning > span { background: var(--warning); color: #0b0b0b; }
  .badge.critical > span { background: var(--critical); } .badge.neutral > span { background: var(--neutral); }
  .progress { height: 6px; border-radius: 3px; background: var(--grid); margin-top: 8px; overflow: hidden; }
  .progress > div { height: 100%; background: var(--s1); border-radius: 3px; }
  section { margin-top: 16px; }
  .caption { color: var(--ink-2); font-size: 13px; margin: 0 0 12px; }
  .notice { font-size: 13px; color: var(--ink-2); border: 1px solid var(--border); border-radius: 8px;
            padding: 8px 12px; margin-bottom: 12px; }
  .chart { position: relative; height: 340px; }
  .empty { color: var(--muted); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 12px;
          font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--grid); }
  th:first-child, td:first-child { text-align: left; } thead th { color: var(--ink-2); font-weight: 600; }
  .table-wrap { overflow-x: auto; }
  details summary { cursor: pointer; color: var(--ink-2); font-size: 13px; margin-top: 12px; }
  footer { color: var(--muted); font-size: 12px; margin-top: 24px; }
  @media (max-width: 600px) { .chart { height: 260px; } table { font-size: 12px; } th, td { padding: 4px; } }
</style>
</head>
<body>
<main>
  <header><h1>AI Trading Lab</h1><span class="generated">Generated __GENERATED__ · read-only snapshot</span></header>
  <div class="tiles">__TILES__</div>
  <section><h2>Forward test</h2>__FORWARD__</section>
  <section><h2>Walk-forward backtest</h2>__BACKTEST__</section>
  <footer>Signal-quality research, not trading advice. Regenerate with: uv run python scripts/build_dashboard.py</footer>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const DATA = __DATA__;
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const money = v => "$" + Math.round(v).toLocaleString();
const charts = [];

function lineChart(id, curves, styles) {
  const el = document.getElementById(id);
  if (!el || !curves || typeof Chart === "undefined") return;
  const datasets = Object.entries(curves.series).map(([name, values]) => {
    const s = styles[name] || {};
    return { label: name, data: values, borderColor: css(s.color), backgroundColor: css(s.color),
             borderWidth: 2, borderDash: s.dash || [], pointRadius: 0, pointHoverRadius: 4,
             pointHoverBorderWidth: 2, pointHoverBorderColor: css("--surface"), tension: 0, spanGaps: true };
  });
  charts.push(new Chart(el, {
    type: "line",
    data: { labels: curves.labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "top", align: "start",
                  labels: { color: css("--ink-2"), boxWidth: 14, boxHeight: 2, usePointStyle: false } },
        tooltip: { backgroundColor: css("--surface"), titleColor: css("--ink"), bodyColor: css("--ink-2"),
                   borderColor: css("--border"), borderWidth: 1, padding: 10,
                   callbacks: { label: c => ` ${c.dataset.label}: ${money(c.parsed.y)}` } }
      },
      scales: {
        x: { ticks: { color: css("--muted"), maxTicksLimit: el.clientWidth < 500 ? 4 : 8, maxRotation: 0,
                      callback: function(v) { return this.getLabelForValue(v).slice(0, 10); } },
             grid: { display: false }, border: { color: css("--axis") } },
        y: { ticks: { color: css("--muted"), callback: v => money(v) },
             grid: { color: css("--grid") }, border: { display: false } }
      }
    }
  }));
}

function render() {
  charts.splice(0).forEach(c => c.destroy());
  if (DATA.backtest) lineChart("backtestChart",
    { labels: DATA.backtest.labels, series: {
        "Base": DATA.backtest.series.base_seed0,
        "+ EUR/USD + USD/JPY": DATA.backtest.series.plus_combined_seed0,
        "Buy & hold gold": DATA.backtest.series.buy_and_hold } },
    { "Base": { color: "--s1" }, "+ EUR/USD + USD/JPY": { color: "--s2" },
      "Buy & hold gold": { color: "--ref", dash: [6, 4] } });
  if (DATA.forward && DATA.forward.curves) {
    const styles = {}; const order = ["--s1", "--s2", "--s3", "--s4"];
    Object.keys(DATA.forward.curves.series).forEach((name, i) => {
      styles[name] = name.startsWith("Buy") ? { color: "--ref", dash: [6, 4] } : { color: order[i] };
    });
    lineChart("forwardChart", DATA.forward.curves, styles);
  }
}
render();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
let resizeTimer; addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 200); });
</script>
</body>
</html>
"""


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Wrote {OUTPUT}. Open it in your browser.")


if __name__ == "__main__":
    main()
