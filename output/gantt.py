"""
output/gantt.py — Gantt Chart Generator for Denim Washing Production Planner
=============================================================================
Generates a self-contained HTML file with:
  - A fixed top bar with KPI cards and filter controls
  - An interactive Gantt chart (scrollable, zoomable)
  - A detail panel triggered by clicking any bar
  - A full operations table below the chart (original style)

Time model: productive minutes (PM), 00h00-23h59, PPD=1440 min/day.
"""

import json
import os
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Dict, List

PPD       = 1440
DAY_START = 0

CMD_PALETTE = [
    "#1D4ED8","#B91C1C","#15803D","#7C3AED","#C2410C","#0E7490","#9D174D",
    "#3D9970","#6D28D9","#92400E","#1E40AF","#991B1B","#166534","#5B21B6",
    "#78350F","#155E75","#7F1D1D","#14532D","#4C1D95","#713F12","#0369A1",
    "#BE185D","#065F46","#4338CA","#0C4A6E","#831843","#052E16",
    "#312E81","#164E63","#DC2626","#16A34A","#2563EB","#9333EA",
    "#EA580C","#0891B2","#D97706","#10B981","#6366F1","#F43F5E",
]

URGENCE_COLORS = {1: "#DC2626", 2: "#F97316", 3: "#EAB308", 4: "#22C55E", 5: "#10B981"}
DAY_NAMES_FR   = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]


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


def _pm_to_clock(pm: int):
    day = pm // PPD
    off = pm % PPD
    h   = off // 60
    m   = off % 60
    return day, h, m


def _machine_label(raw: str) -> str:
    if not raw or raw in ("?", "non assigne"):
        return raw or "?"
    match = re.match(r"^\d+\s+\((.+)\)$", raw.strip())
    return match.group(1) if match else raw.strip()


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
            "cmd":           r["NumeroCommande"],
            "op":            r["NomOperation"],
            "machine_id":    str(r.get("MachineId", -1)),
            "machine":       _machine_label(r.get("MachineName", "?")),
            "urgence":       r.get("Urgence", 2),
            "quantite":      r.get("Quantite", 0),
            "lot":           r.get("LotSize", r.get("QuantiteLot", "?")),
            "lot_idx":       r.get("LotIdx", 0),
            "nb_lots":       r.get("NbLots", 1),
            "dur_min":       r.get("DureeMinutes", 0),
            "dur_chg":       r.get("TempsChargementMinutes", 0),
            "dur_dch":       r.get("TempsDecharementMinutes", 0),
            "dur_total":     r.get("DureeTotale", r.get("DureeMinutes", 0)),
            "s_pm":          s_pm,
            "e_pm":          e_pm,
            "s_day":         s_day,
            "e_day":         e_day,
            "hstart":        f"{s_h:02d}h{s_m:02d}",
            "hend":          f"{e_h:02d}h{e_m:02d}",
            "date_start":    _working_day(J0, s_day).strftime("%d/%m/%Y"),
            "date_end":      _working_day(J0, e_day).strftime("%d/%m/%Y"),
            "date_export":   r.get("DateExport", ""),
            "color":         cmd_colors[r["NumeroCommande"]],
            "urgence_color": URGENCE_COLORS.get(r.get("Urgence", 2), "#94a3b8"),
        })

    active_ids = sorted(
        {t["machine_id"] for t in tasks if t["machine_id"] not in ("-1", "?")},
        key=lambda x: int(x) if x.isdigit() else 0,
    )
    machine_map: Dict[str, str] = {}
    for t in tasks:
        if t["machine_id"] not in machine_map:
            machine_map[t["machine_id"]] = t["machine"]
    machines = [{"id": mid, "name": machine_map[mid]} for mid in active_ids]

    max_day = max(t["e_day"] for t in tasks) + 2
    days = []
    for d in range(max_day):
        wd = _working_day(J0, d)
        days.append({
            "offset":   d,
            "date_str": wd.strftime("%d/%m/%Y"),
            "day_name": DAY_NAMES_FR[wd.weekday()],
            "is_even":  d % 2 == 0,
        })

    by_cmd: Dict[str, list] = defaultdict(list)
    for t in tasks:
        by_cmd[t["cmd"]].append(t)

    deadline_markers = {}
    n_ok, n_late, late_cmds = 0, 0, []
    for nc, cmd_tasks in by_cmd.items():
        fin_pm   = max(t["e_pm"] for t in cmd_tasks)
        exp_date = cmd_tasks[0]["date_export"]
        if exp_date:
            exp_day = _wd_offset(J0, date.fromisoformat(exp_date))
            exp_pm  = (exp_day + 1) * PPD
            deadline_markers[nc] = {
                "pm":   exp_pm,
                "date": exp_date,
                "late": fin_pm > exp_pm,
            }
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
        "on_time_pct":   round(n_ok / len(by_cmd) * 100) if by_cmd else 0,
    }

    config = {
        "PPD":          PPD,
        "DAY_START":    DAY_START,
        "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }

    return {
        "tasks":            tasks,
        "machines":         machines,
        "days":             days,
        "deadline_markers": deadline_markers,
        "kpis":             kpis,
        "config":           config,
    }


_HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Planning Lavage Denim</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f1f5f9;color:#0f172a;font-size:13px;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:#f1f5f9;}
::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px;}

#topbar{background:#fff;border-bottom:1px solid #e2e8f0;padding:12px 20px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 1px 4px rgba(0,0,0,.06);position:sticky;top:0;z-index:200;}
#topbar h1{font-size:17px;font-weight:800;}
#topbar .sub{font-size:11px;color:#94a3b8;margin-top:2px;}

.fbtn{padding:5px 12px;font-size:11px;font-weight:600;border-radius:6px;border:1px solid #e2e8f0;background:#fff;cursor:pointer;color:#475569;transition:all .15s;}
.fbtn.active{background:#0f172a;color:#fff;border-color:#0f172a;}
.fbtn:hover:not(.active){background:#f8fafc;}

.zoom-btn{width:26px;height:26px;border-radius:6px;border:1px solid #e2e8f0;background:#fff;cursor:pointer;color:#475569;font-size:14px;display:flex;align-items:center;justify-content:center;transition:all .15s;}
.zoom-btn:hover{border-color:#6366f1;color:#6366f1;}
.zoom-lbl{font-size:10px;color:#94a3b8;min-width:28px;text-align:center;font-family:monospace;}

#kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;padding:14px 20px;}
.kpi{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:11px 14px;}
.kpi-lbl{font-size:10px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap;}
.kpi-val{font-size:18px;font-weight:700;margin-top:3px;}
.kpi-bar{height:3px;border-radius:2px;margin-top:8px;background:#e2e8f0;}
.kpi-bar-fill{height:100%;border-radius:2px;}

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

.gbar{position:absolute;overflow:hidden;cursor:pointer;display:flex;flex-direction:column;justify-content:center;padding:0 5px;box-shadow:0 1px 3px rgba(0,0,0,.18);}
.gbar:hover{filter:brightness(1.14);z-index:50!important;}
.gbar.dim{opacity:.06;pointer-events:none;}
.gbar.sel{outline:2px solid #fff;outline-offset:1px;z-index:60!important;}
.bt{font-size:9px;font-weight:700;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.3;}
.bs{font-size:8px;color:rgba(255,255,255,.85);font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.3;}

.mlabel{display:flex;flex-direction:column;justify-content:center;padding:0 12px;border-bottom:1px solid #e2e8f0;cursor:pointer;transition:background .1s;}
.mlabel:hover{background:#eff6ff!important;}
.mlabel.hi{background:#dbeafe!important;}
.ml-name{font-size:12px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.ml-sub{font-size:10px;color:#94a3b8;margin-top:1px;}

#dp{display:none;position:fixed;right:18px;bottom:18px;width:290px;background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:15px;box-shadow:0 8px 30px rgba(0,0,0,.13);z-index:300;}
#dp-bar{height:3px;border-radius:2px;margin-bottom:11px;}
.dp-r{display:flex;justify-content:space-between;gap:8px;font-size:11px;margin-bottom:5px;}
.dp-k{color:#94a3b8;flex-shrink:0;}
.dp-v{color:#0f172a;font-weight:600;text-align:right;font-family:monospace;}
.dp-sep{border:none;border-top:1px solid #f1f5f9;margin:7px 0;}
.dp-badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:10px;font-weight:800;}

.section-hd{padding:0 20px 8px;display:flex;justify-content:space-between;align-items:center;}
.section-title{font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.06em;}
.section-meta{font-size:11px;color:#94a3b8;}

#tbl-wrap{margin:0 20px 24px;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.05);}
table{width:100%;border-collapse:collapse;background:#fff;font-size:11px;}
thead tr{background:#f8fafc;border-bottom:2px solid #e2e8f0;}
th{padding:9px 12px;text-align:left;font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap;}
.tr td{padding:7px 12px;border-bottom:1px solid #f1f5f9;}
.tr:last-child td{border-bottom:none;}
.tr:hover td{background:#f8fafc;}
.tr.sel td{background:#eff6ff!important;}
.tr.dim{display:none;}

footer{padding:10px 20px;background:#fff;border-top:1px solid #e2e8f0;font-size:10px;font-family:monospace;color:#94a3b8;text-align:center;}
</style>
</head>
<body>

<div id="topbar">
  <div>
    <h1>Planning Lavage Denim</h1>
    <div class="sub" id="sub"></div>
  </div>
  <div style="display:flex;gap:5px;align-items:center;flex-wrap:wrap;">
    <span style="font-size:10px;font-weight:700;color:#94a3b8;margin-right:2px;">Urgence :</span>
    <button class="fbtn active" data-g="u" data-v="" onclick="filtU(this)">Tous</button>
    <button class="fbtn" data-g="u" data-v="1" onclick="filtU(this)" style="border-left:3px solid #DC2626;">1</button>
    <button class="fbtn" data-g="u" data-v="2" onclick="filtU(this)" style="border-left:3px solid #F97316;">2</button>
    <button class="fbtn" data-g="u" data-v="3" onclick="filtU(this)" style="border-left:3px solid #EAB308;">3</button>
    <button class="fbtn" data-g="u" data-v="4" onclick="filtU(this)" style="border-left:3px solid #22C55E;">4</button>
    <button class="fbtn" data-g="u" data-v="5" onclick="filtU(this)" style="border-left:3px solid #10B981;">5</button>
    <button class="fbtn" onclick="resetAll()" style="color:#64748b;">Reset</button>
    <div style="display:flex;gap:3px;align-items:center;margin-left:6px;">
      <button class="zoom-btn" onclick="zoom(-0.3)">-</button>
      <span class="zoom-lbl" id="zoom-lbl">1.6x</span>
      <button class="zoom-btn" onclick="zoom(0.3)">+</button>
    </div>
  </div>
</div>

<div id="kpis"></div>

<div class="section-hd" style="padding-top:0;margin-top:0;">
  <span class="section-title">Diagramme de Gantt</span>
  <span class="section-meta" id="gantt-meta"></span>
</div>
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

<div class="section-hd">
  <span class="section-title">Operations planifiees</span>
  <span class="section-meta" id="tbl-cnt"></span>
</div>
<div id="tbl-wrap"><div style="overflow-x:auto;"><table>
  <thead><tr>
    <th>Machine</th><th>Commande</th><th>Operation</th>
    <th>Debut</th><th>Fin</th><th>Charg.</th><th>Cycle</th><th>Decharg.</th>
    <th>Pieces</th><th>Qte totale</th>
  </tr></thead>
  <tbody id="tbody"></tbody>
</table></div></div>

<div id="dp">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <span style="font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.06em;">Detail tache</span>
    <button onclick="closeDp()" style="font-size:20px;line-height:1;color:#94a3b8;background:none;border:none;cursor:pointer;">&times;</button>
  </div>
  <div id="dp-bar"></div>
  <div id="dp-body"></div>
</div>

<footer id="foot"></footer>

<script>
const DATA = __PAYLOAD__;

const PPD      = DATA.config.PPD;
const DAY_STRT = DATA.config.DAY_START;
let   PX       = 1.6;
const LBL_W    = 185;
const ROW_H    = 40;
const BAR_H    = 26;
const HDR_DAY  = 28;
const HDR_SEG  = 24;
const HDR_H    = HDR_DAY + HDR_SEG;

function dayW()  { return PPD * PX; }
function pmPx(pm){ return Math.floor(pm/PPD)*dayW() + (pm%PPD)*PX; }

let F = {u:'', mid:''};

function buildKpis() {
  const k = DATA.kpis;
  const pct = k.on_time_pct;
  const cards = [
    ['Debut',           k.debut,                        '#2563EB'],
    ['Fin',             k.fin,                          '#0f172a'],
    ['Commandes',       k.n_cmds,                       '#0f172a'],
    ['Dans les delais', k.n_ok + ' (' + pct + '%)',     '#16A34A'],
    ['En retard',       k.n_late,                       k.n_late ? '#DC2626' : '#16A34A'],
    ['Machines',        k.n_machines,                   '#0891B2'],
    ['Makespan',        k.makespan_days + 'j',          '#EA580C'],
    ['Charge',          k.total_h + 'h',                '#475569'],
  ];
  document.getElementById('kpis').innerHTML = cards.map(([l,v,c]) =>
    `<div class="kpi"><div class="kpi-lbl">${l}</div><div class="kpi-val" style="color:${c};">${v}</div></div>`
  ).join('');
}

function buildHeader() {
  const days = DATA.days;
  const W    = days.length * dayW();
  const hi   = document.getElementById('g-head-inner');
  const hc   = document.getElementById('g-head-corner');
  hc.style.width = LBL_W + 'px';
  hi.style.cssText = `width:${W}px;height:${HDR_H}px;position:relative;`;

  let h = '';
  days.forEach((d, i) => {
    const x  = i * dayW();
    const bg = d.is_even ? '#f8fafc' : '#eef2f7';

    h += `<div style="position:absolute;left:${x.toFixed(1)}px;top:0;width:${dayW().toFixed(1)}px;height:${HDR_DAY}px;
      background:${bg};border-right:2px solid #94a3b8;
      display:flex;align-items:center;justify-content:center;gap:6px;">
      <span style="font-size:11px;font-weight:700;color:#1e293b;">${d.date_str}</span>
      <span style="font-size:10px;color:#64748b;background:#e2e8f0;padding:1px 5px;border-radius:3px;">${d.day_name}</span>
    </div>`;

    const t = HDR_DAY;
    h += `<div style="position:absolute;left:${x.toFixed(1)}px;top:${t}px;width:${dayW().toFixed(1)}px;height:${HDR_SEG}px;
      background:#fefce8;border-right:2px solid #94a3b8;
      display:flex;align-items:center;justify-content:center;">
      <span style="font-size:9px;font-weight:600;color:#92400e;">00h-24h</span></div>`;

    for (let hr = 0; hr <= 24; hr++) {
      const wallH = DAY_STRT + hr;
      const dispH = wallH >= 24 ? wallH - 24 : wallH;
      const tx = x + hr * 60 * PX;
      const isEdge  = (hr===0||hr===24);
      const isMajor = (hr%2===0);
      h += `<div style="position:absolute;left:${tx.toFixed(1)}px;top:${t}px;width:1px;
        height:${isEdge?HDR_SEG:isMajor?HDR_SEG*.6:HDR_SEG*.3}px;
        background:${isEdge?'#94a3b8':isMajor?'#cbd5e1':'#e2e8f0'};pointer-events:none;"></div>`;
      if (isMajor && !isEdge)
        h += `<div style="position:absolute;left:${tx.toFixed(1)}px;top:${t+1}px;transform:translateX(-50%);font-size:8px;font-family:monospace;color:#6b7280;white-space:nowrap;pointer-events:none;">${String(dispH).padStart(2,'0')}h</div>`;
    }
  });
  hi.innerHTML = h;
}

function buildBody() {
  const machines = DATA.machines;
  const tasks    = DATA.tasks;
  const W        = DATA.days.length * dayW();

  const lblEl = document.getElementById('g-labels');
  const inner = document.getElementById('g-inner');
  lblEl.style.width = LBL_W + 'px';

  const byM = {};
  machines.forEach(m => byM[m.id] = []);
  tasks.forEach(t => { if (byM[t.machine_id]) byM[t.machine_id].push(t); });

  let lHtml='', bHtml='', yOff=0;

  machines.forEach((m, mi) => {
    const mTasks = (byM[m.id]||[]).slice().sort((a,b)=>a.s_pm-b.s_pm);
    const bg     = mi%2===0 ? '#fff' : '#f8fafc';

    const tracks=[], tTrack=[];
    mTasks.forEach(t => {
      const xl=pmPx(t.s_pm), xr=pmPx(t.e_pm);
      let placed=false;
      for(let ti=0;ti<tracks.length;ti++){
        if(xl>=tracks[ti]+2){tracks[ti]=xr;tTrack.push(ti);placed=true;break;}
      }
      if(!placed){tracks.push(xr);tTrack.push(tracks.length-1);}
    });

    const nT=Math.max(tracks.length,1);
    const rH=nT*ROW_H+10;

    lHtml += `<div class="mlabel" data-mid="${m.id}" style="height:${rH}px;background:${bg};"
      onclick="filtM('${m.id}')">
      <div class="ml-name">${m.name}</div>
      <div class="ml-sub">${mTasks.length}&nbsp;op.</div>
    </div>`;

    bHtml += `<div style="position:absolute;left:0;top:${yOff}px;width:${W.toFixed(1)}px;height:${rH}px;background:${bg};border-bottom:1px solid #e2e8f0;">`;

    DATA.days.forEach((d,di)=>{
      const dx=di*dayW();
      bHtml += `<div style="position:absolute;left:${(dx+dayW()).toFixed(1)}px;top:0;width:2px;height:100%;background:#cbd5e1;opacity:.4;pointer-events:none;"></div>`;
      for(let hr=2;hr<24;hr+=2){
        const gx=dx+hr*60*PX;
        bHtml += `<div style="position:absolute;left:${gx.toFixed(1)}px;top:0;width:1px;height:100%;background:rgba(100,116,139,.1);pointer-events:none;"></div>`;
      }
    });

    mTasks.forEach((t,ti)=>{
      const track  = tTrack[ti];
      const barTop = 5+track*ROW_H+Math.floor((ROW_H-BAR_H)/2);
      const xLeft  = pmPx(t.s_pm);
      const barW   = Math.max(pmPx(t.e_pm)-xLeft,8);
      const tip    = `${t.cmd} | ${t.op} | ${t.hstart}->${t.hend} | chg=${t.dur_chg}+cyc=${t.dur_min}+dch=${t.dur_dch}min | ${t.lot}pcs`;

      bHtml += `<div class="gbar"
        data-cmd="${t.cmd}" data-op="${t.op}" data-mid="${m.id}" data-urg="${t.urgence}"
        onclick="selBar(this)"
        style="left:${xLeft.toFixed(1)}px;top:${barTop}px;width:${barW.toFixed(1)}px;height:${BAR_H}px;
               background:${t.color};border-radius:4px;border-left:4px solid ${t.urgence_color};z-index:10;"
        title="${tip}">
        <div class="bt">${t.cmd} &middot; ${t.op}</div>
        <div class="bs">${t.hstart}->${t.hend} &middot; ${t.lot}pcs</div>
      </div>`;
    });

    bHtml += '</div>';
    yOff += rH;
  });

  lblEl.innerHTML = lHtml;
  inner.style.cssText = `width:${W.toFixed(1)}px;height:${yOff}px;position:relative;`;
  inner.innerHTML = bHtml;
  document.getElementById('gantt-meta').textContent =
    `${machines.length} machines | ${tasks.length} barres`;
}

function buildTable() {
  const tasks = DATA.tasks.slice().sort((a,b)=>a.s_pm-b.s_pm||(a.machine_id>b.machine_id?1:-1));
  document.getElementById('tbl-cnt').textContent = tasks.length + ' operations';
  document.getElementById('tbody').innerHTML = tasks.map(t =>
    `<tr class="tr" data-cmd="${t.cmd}" data-op="${t.op}" data-mid="${t.machine_id}" data-urg="${t.urgence}">
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
    </tr>`
  ).join('');
}

function initSync() {
  const sc  = document.getElementById('g-scroll');
  const hi  = document.getElementById('g-head-inner');
  const lbl = document.getElementById('g-labels');
  sc.addEventListener('scroll', () => {
    hi.style.transform = `translateX(${-sc.scrollLeft}px)`;
    lbl.scrollTop      = sc.scrollTop;
  });
}

function zoom(delta) {
  const sc = document.getElementById('g-scroll');
  const ratio = sc.scrollLeft / (sc.scrollWidth || 1);
  PX = Math.max(0.4, Math.min(8, PX + delta));
  document.getElementById('zoom-lbl').textContent = PX.toFixed(1) + 'x';
  buildHeader();
  buildBody();
  sc.scrollLeft = ratio * sc.scrollWidth;
}

function filtU(btn) {
  F.u = btn.dataset.v;
  document.querySelectorAll('.fbtn[data-g="u"]').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyF();
}
function filtM(mid) {
  F.mid = F.mid===mid ? '' : mid;
  document.querySelectorAll('.mlabel').forEach(l =>
    l.classList.toggle('hi', l.dataset.mid===F.mid && F.mid!=='')
  );
  applyF();
}
function applyF() {
  document.querySelectorAll('.gbar').forEach(b => {
    b.classList.toggle('dim', !(!F.u||b.dataset.urg===F.u)||!(!F.mid||b.dataset.mid===F.mid));
  });
  document.querySelectorAll('.tr').forEach(r => {
    r.classList.toggle('dim', !(!F.u||r.dataset.urg===F.u)||!(!F.mid||r.dataset.mid===F.mid));
  });
}
function resetAll() {
  F={u:'',mid:''};
  document.querySelectorAll('.fbtn').forEach(b=>b.classList.remove('active'));
  document.querySelector('.fbtn[data-v=""]').classList.add('active');
  document.querySelectorAll('.mlabel').forEach(l=>l.classList.remove('hi'));
  applyF();
}

let selB=null, selR=null;
function selBar(bar) {
  if(selB) selB.classList.remove('sel');
  if(selR) selR.classList.remove('sel');
  bar.classList.add('sel'); selB=bar;
  const d=bar.dataset;
  const t=DATA.tasks.find(x=>x.cmd===d.cmd&&x.op===d.op&&x.machine_id===d.mid);
  if(!t) return;

  const dm=DATA.deadline_markers[t.cmd];
  const dls=dm ? dm.date+(dm.late?' [RETARD]':' [OK]') : 'N/A';
  const uc={'1':'#DC2626','2':'#F97316','3':'#EAB308','4':'#22C55E','5':'#10B981'}[String(t.urgence)]||'#94a3b8';

  document.getElementById('dp-bar').style.background = t.color;
  document.getElementById('dp-body').innerHTML = [
    ['Commande',      t.cmd],
    ['Operation',     t.op],
    ['Machine',       t.machine],
    ['Debut',         t.date_start+' '+t.hstart],
    ['Fin',           t.date_end+' '+t.hend],
    ['Export',        dls],
    null,
    ['Chargement',    t.dur_chg+' min'],
    ['Cycle machine', t.dur_min+' min'],
    ['Dechargement',  t.dur_dch+' min'],
    ['Total lot',     t.dur_total+' min'],
    null,
    ['Pieces (lot)',  t.lot+' pcs'],
    ['Lot',           (t.lot_idx+1)+' / '+t.nb_lots],
    ['Urgence',       t.urgence],
  ].map(row=>{
    if(!row) return `<hr class="dp-sep">`;
    const isUrg=row[0]==='Urgence';
    const vHtml=isUrg
      ? `<span class="dp-badge" style="background:${uc}22;color:${uc};">${row[1]}</span>`
      : `<span class="dp-v">${row[1]}</span>`;
    return `<div class="dp-r"><span class="dp-k">${row[0]}</span>${vHtml}</div>`;
  }).join('');

  document.getElementById('dp').style.display='block';
  document.querySelectorAll('.tr').forEach(r=>{
    const match=r.dataset.cmd===d.cmd&&r.dataset.op===d.op&&r.dataset.mid===d.mid;
    r.classList.toggle('sel',match);
    if(match){selR=r;r.scrollIntoView({behavior:'smooth',block:'nearest'});}
  });
}
function closeDp() {
  document.getElementById('dp').style.display='none';
  if(selB) selB.classList.remove('sel');
  if(selR) selR.classList.remove('sel');
  selB=selR=null;
}

(function(){
  const c=DATA.config, k=DATA.kpis;
  document.getElementById('sub').textContent=
    `CP-SAT | ${k.n_cmds} commandes | ${k.n_machines} machines | 00h-24h (24h/jour) | ${c.generated_at}`;
  document.getElementById('foot').textContent=
    `Planning Lavage Denim | CP-SAT | horaires 00h-24h | ${k.n_machines} machines | ${k.makespan_days} jour(s) | ${c.generated_at}`;
  buildKpis();
  buildHeader();
  buildBody();
  buildTable();
  initSync();
  if(k.late_cmds.length) console.warn('En retard:', k.late_cmds);
})();
</script>
</body>
</html>"""


def generate_gantt(
    results: List[Dict],
    base_date: str = None,
    output_path: str = "output/gantt_chart.html",
) -> None:
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

    payload      = _prepare(results, J0)
    payload_json = json.dumps(payload, ensure_ascii=False)
    html         = _HTML.replace("__PAYLOAD__", payload_json)

    out = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Gantt saved: {out}")