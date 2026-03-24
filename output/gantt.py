"""
output/gantt.py — Gantt Planning Lavage Denim
==============================================
Time model: productive minutes (PM)
  PPD  = 960 PM/day  (08h00 → 00h00, 16 h/day, no lunch break)
  PM 0   = 08h00 day 0
  PM 959 = 23h59 day 0
  PM 960 = 08h00 day 1
"""

import os
import json
import re
from datetime import date, timedelta, datetime
from collections import defaultdict
from typing import List, Dict

# ── Time model constants (must match solver) ──────────────────────────────────
PPD      = 960   # 16 h × 60
DAY_START = 8    # 08h00

# Colour palettes
CMD_PALETTE = [
    "#1D4ED8","#B91C1C","#15803D","#7C3AED","#C2410C","#0E7490","#9D174D",
    "#3D9970","#6D28D9","#92400E","#1E40AF","#991B1B","#166534","#5B21B6",
    "#78350F","#155E75","#7F1D1D","#14532D","#4C1D95","#713F12","#0369A1",
    "#BE185D","#065F46","#4338CA","#0C4A6E","#831843","#052E16",
    "#312E81","#164E63","#DC2626","#16A34A","#2563EB","#9333EA",
    "#EA580C","#0891B2","#D97706","#10B981","#6366F1","#F43F5E",
]
URGENCE_LABELS = {1: "Urgent", 2: "Haute", 3: "Normal", 4: "Basse", 5: "Flexible"}
URGENCE_COLORS = {1: "#DC2626", 2: "#F97316", 3: "#EAB308", 4: "#22C55E", 5: "#10B981"}
DAY_NAMES_FR   = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]


# ── Calendar helpers ──────────────────────────────────────────────────────────

def _working_day(start: date, offset: int) -> date:
    if offset <= 0:
        return start
    d, n = start, 0
    while n < offset:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return d


def _wd_offset(start: date, target: date) -> int:
    if target <= start:
        return 0
    d, count = start, 0
    while d < target:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return count


# ── PM helpers ────────────────────────────────────────────────────────────────

def _pm_to_clock(pm: int):
    """pm → (day_offset, hour, minute)  [08h00-based day, no lunch]"""
    day = pm // PPD
    off = pm % PPD
    h   = DAY_START + off // 60
    m   = off % 60
    if h >= 24:
        h -= 24
    return day, h, m


def _machine_label(raw: str) -> str:
    if not raw or raw in ("?", "non assigné"):
        return raw or "?"
    match = re.match(r'^\d+\s+\((.+)\)$', raw.strip())
    return match.group(1) if match else raw.strip()


# ── Data preparation ──────────────────────────────────────────────────────────

def _prepare(results: List[Dict], J0: date) -> dict:
    cmd_list   = sorted({t["NumeroCommande"] for t in results})
    cmd_colors = {c: CMD_PALETTE[i % len(CMD_PALETTE)] for i, c in enumerate(cmd_list)}

    tasks = []
    for r in results:
        s_pm = r["StartPM"]
        e_pm = r["EndPM"]
        s_day, s_h, s_m = _pm_to_clock(s_pm)
        e_day, e_h, e_m = _pm_to_clock(e_pm)

        tasks.append({
            "cmd":               r["NumeroCommande"],
            "short_cmd":         r["NumeroCommande"].replace("CMD-2026-", "#"),
            "op":                r["NomOperation"],
            "machine_id":        str(r.get("MachineId", -1)),
            "machine":           _machine_label(r.get("MachineName", "?")),
            "urgence":           r.get("Urgence", 2),
            "quantite":          r.get("Quantite", 0),
            "lot":               r.get("LotSize", r.get("QuantiteLot", "?")),
            "nb_cyc":            r.get("NbCycles", 1),
            "lot_idx":           r.get("LotIdx", 0),
            "nb_lots":           r.get("NbLots", 1),
            "dur_min":           r.get("DureeMinutes", 0),
            "dur_chg":           r.get("TempsChargementMinutes", 0),
            "dur_dch":           r.get("TempsDecharementMinutes", 0),
            "dur_total":         r.get("DureeTotale", r.get("DureeMinutes", 0)),
            "s_pm":              s_pm,
            "e_pm":              e_pm,
            "s_day":             s_day,
            "e_day":             e_day,
            "hstart":            f"{s_h:02d}h{s_m:02d}",
            "hend":              f"{e_h:02d}h{e_m:02d}",
            "date_start":        _working_day(J0, s_day).strftime("%d/%m/%Y"),
            "date_end":          _working_day(J0, e_day).strftime("%d/%m/%Y"),
            "date_export":       r.get("DateExport", ""),
            "color":             cmd_colors[r["NumeroCommande"]],
            "urgence_color":     URGENCE_COLORS.get(r.get("Urgence", 2), "#94a3b8"),
            "urgence_label":     URGENCE_LABELS.get(r.get("Urgence", 2), "?"),
        })

    # Machines — ordered by numeric ID
    active_ids  = sorted(
        {t["machine_id"] for t in tasks if t["machine_id"] not in ("-1", "?")},
        key=lambda x: int(x) if x.isdigit() else 0,
    )
    machine_map: Dict[str, str] = {}
    for t in tasks:
        if t["machine_id"] not in machine_map:
            machine_map[t["machine_id"]] = t["machine"]
    machines = [{"id": mid, "name": machine_map[mid]} for mid in active_ids]

    # Days
    max_day = max(t["e_day"] for t in tasks) + 2
    days = []
    for d in range(max_day):
        wd = _working_day(J0, d)
        days.append({
            "offset":   d,
            "date_str": wd.strftime("%d/%m/%Y"),
            "day_name": DAY_NAMES_FR[wd.weekday()],
        })

    # KPIs
    by_cmd: Dict[str, list] = defaultdict(list)
    for t in tasks:
        by_cmd[t["cmd"]].append(t)

    n_ok, n_late, late_cmds = 0, 0, []
    for nc, cmd_tasks in by_cmd.items():
        fin_pm   = max(t["e_pm"] for t in cmd_tasks)
        exp_date = cmd_tasks[0]["date_export"]
        if exp_date:
            exp_day = _wd_offset(J0, date.fromisoformat(exp_date))
            exp_pm  = (exp_day + 1) * PPD
            if fin_pm <= exp_pm:
                n_ok += 1
            else:
                n_late += 1
                late_cmds.append(nc)
        else:
            n_ok += 1

    ms_pm  = max(t["e_pm"] for t in tasks)
    ms_day = _pm_to_clock(ms_pm)[0]

    kpis = {
        "debut":         J0.strftime("%d/%m/%Y"),
        "fin":           _working_day(J0, ms_day).strftime("%d/%m/%Y"),
        "n_cmds":        len(by_cmd),
        "n_ok":          n_ok,
        "n_late":        n_late,
        "late_cmds":     late_cmds,
        "n_machines":    len(machines),
        "makespan_days": ms_day,
        "total_h":       round(sum(t["e_pm"] - t["s_pm"] for t in tasks) / 60),
    }

    config = {
        "PPD":          PPD,
        "DAY_START":    DAY_START,
        "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }

    return {"tasks": tasks, "machines": machines, "days": days,
            "kpis": kpis, "config": config}


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Planning Lavage Denim</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f1f5f9;color:#0f172a;font-size:13px;}

#topbar{background:#fff;border-bottom:1px solid #e2e8f0;padding:12px 20px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 1px 4px rgba(0,0,0,.06);position:sticky;top:0;z-index:200;}
#topbar h1{font-size:17px;font-weight:800;}
#topbar .sub{font-size:11px;color:#94a3b8;margin-top:2px;}

.fbtn{padding:5px 12px;font-size:11px;font-weight:600;border-radius:6px;border:1px solid #e2e8f0;background:#fff;cursor:pointer;color:#475569;transition:all .15s;}
.fbtn.active{background:#0f172a;color:#fff;border-color:#0f172a;}
.fbtn:hover:not(.active){background:#f8fafc;}

#kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;padding:14px 20px;}
.kpi{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:11px 14px;}
.kpi-lbl{font-size:10px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap;}
.kpi-val{font-size:18px;font-weight:700;margin-top:3px;}

#gantt-wrap{margin:0 20px 16px;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,.06);background:#fff;}

#g-head{display:flex;border-bottom:2px solid #94a3b8;background:#f8fafc;flex-shrink:0;}
#g-head-corner{flex-shrink:0;border-right:2px solid #94a3b8;display:flex;align-items:center;padding:0 12px;}
#g-head-corner span{font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.06em;}
#g-head-scroll{flex:1;overflow:hidden;}
#g-head-inner{position:relative;}

#g-body-wrap{display:flex;max-height:560px;}
#g-labels{flex-shrink:0;overflow:hidden;border-right:2px solid #e2e8f0;}
#g-scroll{flex:1;overflow:auto;}
#g-inner{position:relative;}

.gbar{position:absolute;overflow:hidden;cursor:pointer;
  display:flex;flex-direction:column;justify-content:center;padding:0 5px;
  box-shadow:0 1px 3px rgba(0,0,0,.18);}
.gbar:hover{filter:brightness(1.14);z-index:50!important;}
.gbar.dim{opacity:.06;pointer-events:none;}
.gbar.sel{outline:2px solid #fff;outline-offset:1px;z-index:60!important;}
.bt{font-size:9px;font-weight:700;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.3;}
.bs{font-size:8px;color:rgba(255,255,255,.85);font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.3;}

.mlabel{display:flex;flex-direction:column;justify-content:center;padding:0 12px;
  border-bottom:1px solid #e2e8f0;cursor:pointer;transition:background .1s;}
.mlabel:hover{background:#eff6ff!important;}
.mlabel.hi{background:#dbeafe!important;}
.ml-name{font-size:12px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.ml-sub{font-size:10px;color:#94a3b8;margin-top:1px;}

#dp{display:none;position:fixed;right:18px;bottom:18px;width:290px;
  background:#fff;border:1px solid #e2e8f0;border-radius:12px;
  padding:15px;box-shadow:0 8px 30px rgba(0,0,0,.13);z-index:300;}
#dp-bar{height:3px;border-radius:2px;margin-bottom:11px;}
.dp-r{display:flex;justify-content:space-between;gap:8px;font-size:11px;margin-bottom:5px;}
.dp-k{color:#94a3b8;flex-shrink:0;}
.dp-v{color:#0f172a;font-weight:600;text-align:right;font-family:monospace;}
.dp-sep{border:none;border-top:1px solid #f1f5f9;margin:7px 0;}

#tbl-wrap{margin:0 20px 24px;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.05);}
table{width:100%;border-collapse:collapse;background:#fff;font-size:11px;}
thead tr{background:#f8fafc;border-bottom:2px solid #e2e8f0;}
th{padding:9px 12px;text-align:left;font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap;}
.tr td{padding:7px 12px;border-bottom:1px solid #f1f5f9;}
.tr:last-child td{border-bottom:none;}
.tr:hover td{background:#f8fafc;}
.tr.sel td{background:#eff6ff!important;}
.tr.dim{display:none;}

footer{padding:10px 20px;background:#fff;border-top:1px solid #e2e8f0;
  font-size:10px;font-family:monospace;color:#94a3b8;text-align:center;}
</style>
</head>
<body>

<div id="topbar">
  <div><h1>Planning Lavage Denim</h1><div class="sub" id="sub"></div></div>
  <div style="display:flex;gap:6px;align-items:center;">
    <button class="fbtn active" data-g="u" data-v="" onclick="filt(this)">Toutes</button>
    <button class="fbtn" data-g="u" data-v="1" onclick="filt(this)" style="border-left:3px solid #EF4444;">Urgent</button>
    <button class="fbtn" data-g="u" data-v="2" onclick="filt(this)" style="border-left:3px solid #F97316;">Normal</button>
    <button class="fbtn" data-g="u" data-v="3" onclick="filt(this)" style="border-left:3px solid #10B981;">Flexible</button>
    <button class="fbtn" onclick="resetAll()" style="color:#64748b;">↺ Reset</button>
  </div>
</div>

<div id="kpis"></div>

<div id="gantt-wrap">
  <div id="g-head">
    <div id="g-head-corner"><span>Machine</span></div>
    <div id="g-head-scroll"><div id="g-head-inner"></div></div>
  </div>
  <div id="g-body-wrap">
    <div id="g-labels"></div>
    <div id="g-scroll"><div id="g-inner"></div></div>
  </div>
</div>

<div style="padding:0 20px 8px;display:flex;justify-content:space-between;align-items:center;">
  <span style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.06em;">Opérations planifiées</span>
  <span id="tbl-cnt" style="font-size:11px;color:#94a3b8;"></span>
</div>
<div id="tbl-wrap"><div style="overflow-x:auto;"><table>
  <thead><tr>
    <th>Machine</th><th>Commande</th><th>Opération</th>
    <th>Début</th><th>Fin</th><th>Charg.</th><th>Cycle</th><th>Décharg.</th>
    <th>Pièces</th><th>Qté totale</th>
  </tr></thead>
  <tbody id="tbody"></tbody>
</table></div></div>

<div id="dp">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <span style="font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.06em;">Détail tâche</span>
    <button onclick="closeDp()" style="font-size:20px;line-height:1;color:#94a3b8;background:none;border:none;cursor:pointer;">&times;</button>
  </div>
  <div id="dp-bar"></div>
  <div id="dp-body"></div>
</div>

<footer id="foot"></footer>

<script>
// ── DATA (injected by Python) ──────────────────────────────────────
const DATA = __PAYLOAD__;

// ── Layout constants ───────────────────────────────────────────────
const PPD      = DATA.config.PPD;        // 960
const DAY_H    = PPD / 60;               // 16 hours per day
const PX       = 1.6;                    // px per productive minute (wider = 960*1.6=1536px/day)
const DAY_W    = PPD * PX;              // total pixels per working day
const LBL_W    = 185;
const ROW_H    = 40;
const BAR_H    = 26;
const HDR_DAY  = 28;
const HDR_SEG  = 24;
const HDR_H    = HDR_DAY + HDR_SEG;
const DAY_START = DATA.config.DAY_START; // 8

// ── Coordinate helpers ─────────────────────────────────────────────
function px(pm) {
  const day = Math.floor(pm / PPD);
  const off = pm % PPD;
  return day * DAY_W + off * PX;
}

function pmClock(pm) {
  const day = Math.floor(pm / PPD);
  const off = pm % PPD;
  let h = DAY_START + Math.floor(off / 60);
  const m = off % 60;
  if (h >= 24) h -= 24;
  return {day, h, m};
}

// ── Build header ───────────────────────────────────────────────────
function buildHeader() {
  const days = DATA.days;
  const W    = days.length * DAY_W;
  const hi   = document.getElementById('g-head-inner');
  const hc   = document.getElementById('g-head-corner');
  hc.style.width = LBL_W + 'px';
  hi.style.cssText = `width:${W}px;height:${HDR_H}px;position:relative;`;

  let h = '';
  days.forEach((d, i) => {
    const x  = i * DAY_W;
    const bg = i%2===0 ? '#f8fafc' : '#eef2f7';

    // Day label row
    h += `<div style="position:absolute;left:${x}px;top:0;width:${DAY_W}px;height:${HDR_DAY}px;
      background:${bg};border-right:2px solid #94a3b8;
      display:flex;align-items:center;justify-content:center;gap:6px;">
      <span style="font-size:11px;font-weight:700;color:#1e293b;">${d.date_str}</span>
      <span style="font-size:10px;color:#64748b;background:#e2e8f0;padding:1px 5px;border-radius:3px;">${d.day_name}</span>
    </div>`;

    const t = HDR_DAY;

    // Single continuous day block (08h → 00h)
    h += `<div style="position:absolute;left:${x}px;top:${t}px;width:${DAY_W}px;height:${HDR_SEG}px;
      background:#fefce8;border-right:2px solid #94a3b8;
      display:flex;align-items:center;justify-content:center;">
      <span style="font-size:9px;font-weight:600;color:#92400e;">08h → 00h</span></div>`;

    // Hour ticks: 08h to 00h (every 2h for readability at this scale)
    for (let hr = 0; hr <= 16; hr++) {
      const wallH = DAY_START + hr;
      const displayH = wallH >= 24 ? wallH - 24 : wallH;
      const tx = x + hr * 60 * PX;
      const isEdge = (hr === 0 || hr === 16);
      const isMajor = (hr % 2 === 0);
      const lbl = `${String(displayH).padStart(2,'0')}h`;
      h += `<div style="position:absolute;left:${tx.toFixed(1)}px;top:${t}px;width:1px;
        height:${isEdge ? HDR_SEG : isMajor ? HDR_SEG*0.6 : HDR_SEG*0.3}px;
        background:${isEdge ? '#94a3b8' : isMajor ? '#cbd5e1' : '#e2e8f0'};pointer-events:none;"></div>`;
      if (isMajor && !isEdge) {
        h += `<div style="position:absolute;left:${tx.toFixed(1)}px;top:${t+1}px;
          transform:translateX(-50%);font-size:8px;font-family:monospace;color:#6b7280;white-space:nowrap;pointer-events:none;">${lbl}</div>`;
      }
    }
  });

  hi.innerHTML = h;
}

// ── Build body ─────────────────────────────────────────────────────
function buildBody() {
  const machines = DATA.machines;
  const tasks    = DATA.tasks;
  const nDays    = DATA.days.length;
  const W        = nDays * DAY_W;

  const lblEl  = document.getElementById('g-labels');
  const inner  = document.getElementById('g-inner');
  lblEl.style.width = LBL_W + 'px';

  const byM = {};
  machines.forEach(m => byM[m.id] = []);
  tasks.forEach(t => { if (byM[t.machine_id]) byM[t.machine_id].push(t); });

  let lHtml = '', bHtml = '';
  let yOff  = 0;

  machines.forEach((m, mi) => {
    const mTasks = (byM[m.id]||[]).slice().sort((a,b) => a.s_pm - b.s_pm);
    const bg     = mi%2===0 ? '#fff' : '#f8fafc';

    // Track packing
    const BAR_W_PACK = 120;
    const tracks = [];
    const tTrack = [];
    mTasks.forEach(t => {
      const xl = px(t.s_pm);
      const xr = xl + BAR_W_PACK;
      let placed = false;
      for (let ti = 0; ti < tracks.length; ti++) {
        if (xl >= tracks[ti] + 2) { tracks[ti] = xr; tTrack.push(ti); placed = true; break; }
      }
      if (!placed) { tracks.push(xr); tTrack.push(tracks.length - 1); }
    });

    const nT = Math.max(tracks.length, 1);
    const rH = nT * ROW_H + 10;

    // Label
    lHtml += `<div class="mlabel" data-mid="${m.id}" style="height:${rH}px;background:${bg};"
      onclick="filtMach('${m.id}',this)">
      <div class="ml-name">${m.name}</div>
      <div class="ml-sub">${mTasks.length}&nbsp;op(s)</div>
    </div>`;

    // Row background
    bHtml += `<div style="position:absolute;left:0;top:${yOff}px;width:${W}px;height:${rH}px;background:${bg};border-bottom:1px solid #e2e8f0;">`;

    // Day stripes + hour grid lines (no lunch stripe)
    DATA.days.forEach((d, di) => {
      const dx = di * DAY_W;
      const sb = di%2===0 ? 'rgba(254,252,232,.15)' : 'rgba(239,246,255,.15)';
      bHtml += `<div style="position:absolute;left:${dx}px;width:${DAY_W}px;height:100%;background:${sb};pointer-events:none;"></div>`;
      // Day separator
      bHtml += `<div style="position:absolute;left:${dx+DAY_W}px;top:0;width:2px;height:100%;background:#cbd5e1;opacity:.4;pointer-events:none;"></div>`;
      // Hour grid lines every 2h
      for (let hr = 2; hr < 16; hr += 2) {
        const gx = dx + hr * 60 * PX;
        bHtml += `<div style="position:absolute;left:${gx.toFixed(1)}px;top:0;width:1px;height:100%;background:rgba(100,116,139,.1);pointer-events:none;"></div>`;
      }
      // Midnight marker (end of day)
      const midnightX = dx + 16 * 60 * PX;
      bHtml += `<div style="position:absolute;left:${midnightX.toFixed(1)}px;top:0;width:1px;height:100%;background:rgba(100,116,139,.25);pointer-events:none;"></div>`;
    });

    // Task bars
    const BAR_W = 120;
    mTasks.forEach((t, ti) => {
      const track  = tTrack[ti];
      const barTop = 5 + track * ROW_H + Math.floor((ROW_H - BAR_H) / 2);
      const xLeft  = px(t.s_pm);
      // Actual bar width proportional to duration
      const barW   = Math.max((t.e_pm - t.s_pm) * PX, 8);
      const tipText = `${t.cmd} · ${t.op} · ${t.hstart}→${t.hend} · chg=${t.dur_chg}+cyc=${t.dur_min}+dch=${t.dur_dch}min · ${t.lot}pcs`;
      const inner2  = `<div class="bt">${t.short_cmd} · ${t.op}</div>`
                    + `<div class="bs">${t.hstart}→${t.hend} · ${t.lot}pcs</div>`;

      bHtml += `<div class="gbar"
        data-cmd="${t.cmd}" data-op="${t.op}" data-mid="${m.id}" data-urg="${t.urgence}"
        onclick="selBar(this)"
        style="left:${xLeft.toFixed(1)}px;top:${barTop}px;
               width:${barW.toFixed(1)}px;height:${BAR_H}px;
               background:${t.color};border-radius:4px;
               border-left:4px solid ${t.urgence_color};z-index:10;"
        title="${tipText}">${inner2}</div>`;
    });

    bHtml += '</div>';
    yOff  += rH;
  });

  lblEl.innerHTML = lHtml;
  inner.style.cssText = `width:${W}px;height:${yOff}px;position:relative;`;
  inner.innerHTML = bHtml;
}

// ── KPIs ───────────────────────────────────────────────────────────
function buildKpis() {
  const k = DATA.kpis;
  const okR = k.n_cmds ? Math.round(k.n_ok / k.n_cmds * 100) : 0;
  const cards = [
    ['Début',        k.debut,              '#2563EB'],
    ['Fin',          k.fin,                '#0f172a'],
    ['Commandes',    k.n_cmds,             '#0f172a'],
    ['Dans les délais', `${k.n_ok} (${okR}%)`, '#16A34A'],
    ['En retard',    k.n_late,             k.n_late ? '#DC2626' : '#16A34A'],
    ['Machines',     k.n_machines,         '#0891B2'],
    ['Makespan',     `${k.makespan_days}j`,'#EA580C'],
    ['Charge',       `${k.total_h}h`,      '#475569'],
  ];
  document.getElementById('kpis').innerHTML = cards.map(([l,v,c]) =>
    `<div class="kpi"><div class="kpi-lbl">${l}</div><div class="kpi-val" style="color:${c};">${v}</div></div>`
  ).join('');
}

// ── Table ──────────────────────────────────────────────────────────
function buildTable() {
  const tasks = DATA.tasks.slice().sort((a,b) => a.s_pm - b.s_pm || (a.machine_id > b.machine_id ? 1 : -1));
  document.getElementById('tbl-cnt').textContent = tasks.length + ' opérations';
  document.getElementById('tbody').innerHTML = tasks.map(t => {
    return `<tr class="tr" data-cmd="${t.cmd}" data-op="${t.op}" data-mid="${t.machine_id}" data-urg="${t.urgence}">
      <td style="font-weight:600;">${t.machine}</td>
      <td><div style="display:flex;align-items:center;gap:6px;">
        <div style="width:10px;height:10px;border-radius:2px;background:${t.color};border-left:3px solid ${t.urgence_color};flex-shrink:0;"></div>
        <span style="font-family:monospace;font-weight:700;">${t.cmd}</span></div></td>
      <td>${t.op}</td>
      <td style="font-family:monospace;white-space:nowrap;">${t.date_start} ${t.hstart}</td>
      <td style="font-family:monospace;white-space:nowrap;">${t.date_end} ${t.hend}</td>
      <td style="font-family:monospace;color:#0891B2;">${t.dur_chg}min</td>
      <td style="font-family:monospace;color:#475569;">${t.dur_min}min</td>
      <td style="font-family:monospace;color:#EA580C;">${t.dur_dch}min</td>
      <td style="font-family:monospace;color:#64748b;">${t.lot}pcs</td>
      <td style="font-family:monospace;color:#94a3b8;">${t.quantite.toLocaleString()}pcs</td>
    </tr>`;
  }).join('');
}

// ── Scroll sync ────────────────────────────────────────────────────
function initSync() {
  const sc  = document.getElementById('g-scroll');
  const hi  = document.getElementById('g-head-inner');
  const lbl = document.getElementById('g-labels');
  sc.addEventListener('scroll', () => {
    hi.style.transform = `translateX(${-sc.scrollLeft}px)`;
    lbl.scrollTop      = sc.scrollTop;
  });
}

// ── Filters ────────────────────────────────────────────────────────
let F = {u: '', mid: ''};

function filt(btn) {
  F.u = btn.dataset.v;
  document.querySelectorAll('.fbtn[data-g="u"]').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  apply();
}
function filtMach(mid, el) {
  F.mid = F.mid === mid ? '' : mid;
  document.querySelectorAll('.mlabel').forEach(l => l.classList.toggle('hi', l.dataset.mid === F.mid && F.mid !== ''));
  apply();
}
function apply() {
  document.querySelectorAll('.gbar[data-cmd]').forEach(b => {
    const ok = (!F.u || b.dataset.urg === F.u) && (!F.mid || b.dataset.mid === F.mid);
    b.classList.toggle('dim', !ok);
  });
  document.querySelectorAll('.tr').forEach(r => {
    const ok = (!F.u || r.dataset.urg === F.u) && (!F.mid || r.dataset.mid === F.mid);
    r.classList.toggle('dim', !ok);
  });
}
function resetAll() {
  F = {u: '', mid: ''};
  document.querySelectorAll('.fbtn').forEach(b => b.classList.remove('active'));
  document.querySelector('.fbtn[data-v=""]').classList.add('active');
  document.querySelectorAll('.mlabel').forEach(l => l.classList.remove('hi'));
  apply();
}

// ── Detail panel ───────────────────────────────────────────────────
let selB = null, selR = null;

function selBar(bar) {
  if (selB) selB.classList.remove('sel');
  if (selR) selR.classList.remove('sel');
  bar.classList.add('sel'); selB = bar;
  const d = bar.dataset;
  const t = DATA.tasks.find(x => x.cmd === d.cmd && x.op === d.op && x.machine_id === d.mid);
  if (!t) return;
  document.getElementById('dp-bar').style.background = t.color;
  document.getElementById('dp-body').innerHTML = [
    ['Commande',       t.cmd],
    ['Opération',      t.op],
    ['Machine',        t.machine],
    ['Début',          `${t.date_start} ${t.hstart}`],
    ['Fin',            `${t.date_end} ${t.hend}`],
    ['—', null],
    ['Chargement',     `${t.dur_chg} min`],
    ['Cycle machine',  `${t.dur_min} min`],
    ['Déchargement',   `${t.dur_dch} min`],
    ['Total machine',  `${t.dur_total} min`],
    ['—', null],
    ['Pièces (lot)',   `${t.lot} pcs`],
    ['Qté totale cmd', `${t.quantite.toLocaleString()} pcs`],
    ['Urgence',        t.urgence_label],
  ].map(([k, v]) => {
    if (v === null) return `<hr class="dp-sep">`;
    return `<div class="dp-r"><span class="dp-k">${k}</span><span class="dp-v">${v}</span></div>`;
  }).join('');
  document.getElementById('dp').style.display = 'block';
  document.querySelectorAll('.tr').forEach(r => {
    const match = r.dataset.cmd === d.cmd && r.dataset.op === d.op && r.dataset.mid === d.mid;
    r.classList.toggle('sel', match);
    if (match) { selR = r; r.scrollIntoView({behavior:'smooth', block:'nearest'}); }
  });
}
function closeDp() {
  document.getElementById('dp').style.display = 'none';
  if (selB) selB.classList.remove('sel');
  if (selR) selR.classList.remove('sel');
  selB = selR = null;
}

// ── Init ───────────────────────────────────────────────────────────
(function() {
  const c = DATA.config, k = DATA.kpis;
  document.getElementById('sub').textContent =
    `CP-SAT · ${k.n_cmds} commandes · ${k.n_machines} machines · 08h-00h (16h/jour) · ${c.generated_at}`;
  document.getElementById('foot').textContent =
    `Planning Lavage Denim · CP-SAT · horaires 08h-00h · ${k.n_machines} machines · ${k.makespan_days} jour(s) · ${c.generated_at}`;
  buildKpis();
  buildHeader();
  buildBody();
  buildTable();
  initSync();
  if (k.late_cmds.length) console.warn('En retard:', k.late_cmds);
})();
</script>
</body>
</html>"""


def generate_gantt(results, base_date=None, output_path="output/gantt_chart.html"):
    if not results:
        print("No results to render.")
        return

    if base_date:
        J0 = date.fromisoformat(base_date)
    elif results[0].get("DateStart"):
        J0 = date.fromisoformat(results[0]["DateStart"])
    else:
        J0 = date.today()
    while J0.weekday() >= 5:
        J0 += timedelta(days=1)

    payload = _prepare(results, J0)
    payload_json = json.dumps(payload, ensure_ascii=False)

    html = _HTML_TEMPLATE.replace("__PAYLOAD__", payload_json)

    out = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Gantt saved → {out}")