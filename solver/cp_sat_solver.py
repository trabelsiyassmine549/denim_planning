"""
solver/cp_sat_solver.py — Planification Lavage Denim (CP-SAT)
=============================================================
Time model: productive minutes (PM) — 08h00 → 00h00, 960 PM/day.
 
Strategy (two-phase):
  Phase 1 — Greedy machine pre-assignment
    For each (cmd, op), pick the least-loaded available machine.
    This eliminates 82 000+ boolean assignment variables from CP-SAT
    and makes the model trivially small.
 
  Phase 2 — CP-SAT timing only
    One interval task per (cmd, op) on its pre-assigned machine.
    NoOverlap per machine + precedence between ops + tardiness objective.
    ~5 000 tasks, ~5 000 integer vars → solves in seconds.
 
Gantt output: each task is expanded back to per-lot bars post-solve.
"""
 
import math
import time
from collections import defaultdict
from typing import Dict, List
 
from ortools.sat.python import cp_model
 
from utils.data_loader import load_data, validate_data
from utils.time_utils import (
    PPD,
    START_DATE,
    date_to_day_offset,
    date_to_pm,
    pm_to_clock,
    working_day_date,
    fmt_date,
)
 
MAX_SOLVE_SECONDS = 120
RANDOM_SEED       = 42
 
 
# ── Phase 1: greedy load-balanced machine assignment ─────────────────────────
 
def _build_machines_by_op(machines_ok):
    idx = {}
    for m in machines_ok:
        for op in m.operations_list():
            idx.setdefault(op, []).append(m)
    return idx
 
 
def _greedy_assign(commandes, ops_by_recette, machines_by_op):
    """
    For each (cmd, op_idx) assign one machine, balancing total load minutes.
    Returns: assign[(cmd_nc, op_idx)] = Machine object
    """
    machine_load: Dict[int, int] = defaultdict(int)
    assign: Dict[tuple, object]  = {}
 
    # Sort commands: urgent first, then earlier deadline
    sorted_cmds = sorted(
        commandes,
        key=lambda c: (c.Urgence, date_to_day_offset(c.DateExport))
    )
 
    for cmd in sorted_cmds:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for op_idx, op in enumerate(ops):
            avail = machines_by_op.get(op.NomOperation, [])
            if not avail:
                continue
 
            # Pick machine with lowest current load
            best = min(avail, key=lambda m: machine_load[m.Id])
            assign[(cmd.NumeroCommande, op_idx)] = best
 
            # FIX: use the assigned machine's actual capacity (not the global minimum)
            # Using min_cap caused huge lot counts (e.g. 20 lots instead of 4)
            # which serialized all work on one machine, preventing load balancing.
            lot_size  = min(op.QuantiteLot, best.CapaciteMax)
            nb_lots   = math.ceil(cmd.Quantite / lot_size)
            dur_total = op.DureeTotale * nb_lots
            machine_load[best.Id] += dur_total
 
    return assign
 
 
# ── Phase 2: CP-SAT timing solver ────────────────────────────────────────────
 
def solve():
    commandes, machines, ops_by_recette, recettes_by_id = load_data()
    for w in validate_data(commandes, machines, ops_by_recette, recettes_by_id):
        print(f"⚠️  {w}")
 
    machines_ok    = [m for m in machines if m.is_available()]
    machines_by_op = _build_machines_by_op(machines_ok)
 
    print(f"📋 {len(commandes)} commandes | {len(machines_ok)}/{len(machines)} machines OK")
    print(f"⏱  Horaires: 08h00 → 00h00 (16h/jour)  |  PPD = {PPD} min/jour")
 
    # Phase 1
    print("⚙️  Phase 1 — assignation machine (greedy load-balance)...")
    assign = _greedy_assign(commandes, ops_by_recette, machines_by_op)
    print(f"   {len(assign)} tâches pré-assignées")
 
    max_export_day = max(date_to_day_offset(c.DateExport) for c in commandes)
    horizon        = (max_export_day + 10) * PPD
 
    # Phase 2 — build CP-SAT model (timing only)
    model         = cp_model.CpModel()
    task_vars:    Dict[tuple, dict] = {}
    machine_itvs: Dict[int, list]   = {m.Id: [] for m in machines_ok}
 
    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for op_idx, op in enumerate(ops):
            key = (cmd.NumeroCommande, op_idx)
            if key not in assign:
                continue
 
            avail    = machines_by_op.get(op.NomOperation, [])
            # FIX: use the assigned machine's actual capacity for lot sizing,
            # not the global minimum across all machines for this operation.
            assigned_machine = assign[key]
            lot_size = min(op.QuantiteLot, assigned_machine.CapaciteMax)
            nb_lots  = math.ceil(cmd.Quantite / lot_size)
            dur      = op.DureeTotale * nb_lots
 
            s_var = model.NewIntVar(0, horizon - dur, f"s_{cmd.NumeroCommande}_{op_idx}")
            e_var = model.NewIntVar(0, horizon,       f"e_{cmd.NumeroCommande}_{op_idx}")
            model.Add(e_var == s_var + dur)
 
            iv = model.NewIntervalVar(s_var, dur, e_var, f"iv_{cmd.NumeroCommande}_{op_idx}")
            machine_itvs[assign[key].Id].append(iv)
 
            task_vars[key] = {
                "start":                   s_var,
                "end":                     e_var,
                "NomOperation":            op.NomOperation,
                "DureeMinutes":            op.DureeMinutes,
                "TempsChargementMinutes":  op.TempsChargementMinutes,
                "TempsDecharementMinutes": op.TempsDecharementMinutes,
                "DureeTotale":             op.DureeTotale,
                "LotSize":                 lot_size,
                "NbLots":                  nb_lots,
            }
 
    # No overlap per machine
    for m in machines_ok:
        if machine_itvs[m.Id]:
            model.AddNoOverlap(machine_itvs[m.Id])
 
    print(f"📐 {len(task_vars)} tâches | CP-SAT: timing uniquement")
 
    # Inter-operation precedence
    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for i in range(len(ops) - 1):
            k0 = (cmd.NumeroCommande, i)
            k1 = (cmd.NumeroCommande, i + 1)
            if k0 in task_vars and k1 in task_vars:
                model.Add(task_vars[k1]["start"] >= task_vars[k0]["end"])
 
    # Tardiness
    tard_vars, weights = [], []
    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        if not ops:
            continue
        last_key = (cmd.NumeroCommande, len(ops) - 1)
        if last_key not in task_vars:
            continue
        deadline = date_to_pm(cmd.DateExport)
        tard = model.NewIntVar(0, horizon, f"tard_{cmd.NumeroCommande}")
        late = model.NewIntVar(-horizon, horizon, f"late_{cmd.NumeroCommande}")
        model.Add(late == task_vars[last_key]["end"] - deadline)
        model.AddMaxEquality(tard, [late, model.NewConstant(0)])
        tard_vars.append(tard)
        weights.append(10 // cmd.Urgence)
 
    # Makespan
    makespan = model.NewIntVar(0, horizon, "makespan")
    all_ends = [
        task_vars[(cmd.NumeroCommande, len(ops_by_recette.get(cmd.RecetteId, [])) - 1)]["end"]
        for cmd in commandes
        if (cmd.NumeroCommande, len(ops_by_recette.get(cmd.RecetteId, [])) - 1) in task_vars
    ]
    if all_ends:
        model.AddMaxEquality(makespan, all_ends)
 
    # Objective: minimize weighted tardiness then makespan
    TARD_PENALTY = 100_000
    model.Minimize(
        TARD_PENALTY * sum(w * t for w, t in zip(weights, tard_vars))
        + makespan
    )
 
    print(f"\n{'━'*70}")
    print(f"  RÉSOLUTION — limite {MAX_SOLVE_SECONDS}s | seed={RANDOM_SEED}")
    print(f"{'━'*70}")
 
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAX_SOLVE_SECONDS
    solver.parameters.random_seed         = RANDOM_SEED
    # Let CP-SAT auto-detect all CPU cores — do NOT set num_search_workers
    t0     = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0
 
    status_str = {
        cp_model.OPTIMAL:    "✅ OPTIMAL",
        cp_model.FEASIBLE:   "🟡 RÉALISABLE",
        cp_model.INFEASIBLE: "❌ INFAISABLE",
        cp_model.UNKNOWN:    "❓ INCONNU (timeout)",
    }.get(status, f"? code={status}")
    print(f"\n  {status_str}  |  {elapsed:.1f}s")
 
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []
 
    ms     = solver.Value(makespan)
    ms_day = ms // PPD
    print(f"  Makespan : {ms_day} jours ouvrés ({ms} min productives)\n")
    print(f"\n{'='*70}")
    print(f"✅ Planning résolu  |  Makespan : {ms_day} jours ouvrés")
    print(f"📅 Début : {START_DATE}  |  Fin : {fmt_date(ms_day)}")
    print(f"{'='*70}\n")
 
    # ── Extract results — expand tasks back to per-lot bars ───────────────────
    results = []
    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        if not ops:
            continue
        print(f"📦 {cmd.NumeroCommande} | qty={cmd.Quantite} | urgence={cmd.Urgence}")
 
        for op_idx, op in enumerate(ops):
            key = (cmd.NumeroCommande, op_idx)
            tv  = task_vars.get(key)
            if not tv:
                continue
 
            task_start_pm = solver.Value(tv["start"])
            nb_lots       = tv["NbLots"]
            lot_size      = tv["LotSize"]
            dur_one       = op.DureeTotale
            m_assigned    = assign[key]
            assigned_name = f"{m_assigned.Id} ({m_assigned.NomMachine})"
 
            for lot_idx in range(nb_lots):
                s_pm = task_start_pm + lot_idx * dur_one
                e_pm = s_pm + dur_one
                s_day, s_h, s_m = pm_to_clock(s_pm)
                e_day, e_h, e_m = pm_to_clock(e_pm)
                d_start = working_day_date(s_day)
                d_end   = working_day_date(e_day)
                pieces_this_lot = (
                    cmd.Quantite - lot_idx * lot_size
                    if lot_idx == nb_lots - 1 else lot_size
                )
 
                print(f"   [{op_idx+1}.{lot_idx+1}] {op.NomOperation:20s}: "
                      f"{d_start} {s_h:02d}h{s_m:02d}→{e_h:02d}h{e_m:02d} "
                      f"[chg={op.TempsChargementMinutes}+cyc={op.DureeMinutes}"
                      f"+dch={op.TempsDecharementMinutes}] | "
                      f"{pieces_this_lot}pcs | {assigned_name}")
 
                results.append({
                    "NumeroCommande":          cmd.NumeroCommande,
                    "Quantite":                cmd.Quantite,
                    "RecetteId":               cmd.RecetteId,
                    "Urgence":                 cmd.Urgence,
                    "NomOperation":            op.NomOperation,
                    "MachineId":               m_assigned.Id,
                    "MachineName":             assigned_name,
                    "StartPM":                 s_pm,
                    "EndPM":                   e_pm,
                    "DureeMinutes":            op.DureeMinutes,
                    "TempsChargementMinutes":  op.TempsChargementMinutes,
                    "TempsDecharementMinutes": op.TempsDecharementMinutes,
                    "DureeTotale":             op.DureeTotale,
                    "NbCycles":                1,
                    "LotSize":                 pieces_this_lot,
                    "QuantiteLot":             lot_size,
                    "LotIdx":                  lot_idx,
                    "NbLots":                  nb_lots,
                    "DateStart":               d_start.isoformat(),
                    "DateEnd":                 d_end.isoformat(),
                    "DateExport":              cmd.DateExport,
                })
 
        deadline_day = date_to_day_offset(cmd.DateExport)
        last_key     = (cmd.NumeroCommande, len(ops) - 1)
        if last_key in task_vars:
            fin_pm  = solver.Value(task_vars[last_key]["end"])
            fin_day = fin_pm // PPD
            if fin_day > deadline_day:
                print(f"   ⚠️  RETARD {fin_day - deadline_day} jour(s)\n")
            else:
                print(f"   ✅ marge {deadline_day - fin_day} jour(s)\n")
 
    return results
 
 
if __name__ == "__main__":
    solve()