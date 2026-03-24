"""
solver/cp_sat_solver.py — Planification Lavage Denim (CP-SAT)
=============================================================
Time model: productive minutes (PM) — 08h00 → 00h00, 960 PM/day.
No lunch break. No cross-segment restriction on tasks.

Each lot (cycle) of each operation is its own independent CP-SAT task:
  - Duration on machine = TempsChargementMinutes + DureeMinutes + TempsDecharementMinutes
    (the full machine occupation: load → run → unload)
  - Lots of the same (cmd, op) are chained: lot[k+1].start >= lot[k].end
  - Inter-operation precedence: first lot of op[i+1] starts after last lot of op[i]
    (déchargement of op[i] already included in lot[i]'s duration, so no extra gap needed)
  - Machine no-overlap is enforced per lot (using DureeTotale)
  - Each lot bar appears individually on the Gantt
"""

import math
import time
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

MAX_SOLVE_SECONDS = 180
NUM_WORKERS       = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_machines_by_op(machines_ok):
    idx = {}
    for m in machines_ok:
        for op in m.operations_list():
            idx.setdefault(op, []).append(m)
    return idx


# ── Solver ────────────────────────────────────────────────────────────────────

def solve():
    commandes, machines, ops_by_recette, recettes_by_id = load_data()
    for w in validate_data(commandes, machines, ops_by_recette, recettes_by_id):
        print(f"⚠️  {w}")

    machines_ok    = [m for m in machines if m.is_available()]
    machines_by_op = _build_machines_by_op(machines_ok)

    print(f"📋 {len(commandes)} commandes | {len(machines_ok)}/{len(machines)} machines OK")
    print(f"⏱  Horaires: 08h00 → 00h00 (16h/jour)  |  PPD = {PPD} min/jour")

    max_export_day = max(date_to_day_offset(c.DateExport) for c in commandes)
    horizon        = (max_export_day + 10) * PPD

    model        = cp_model.CpModel()
    # lot_vars[(cmd_nc, op_idx, lot_idx)] = {start, end, dur_total, dur_cycle, ...}
    lot_vars:    Dict[tuple, dict] = {}
    # machine_vars[(cmd_nc, op_idx, lot_idx, machine_id)] = BoolVar
    machine_vars: Dict[tuple, object] = {}
    machine_itvs: Dict[int, list]     = {m.Id: [] for m in machines_ok}

    total_lots = 0

    # ── Build one task per lot ────────────────────────────────────────────────
    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for op_idx, op in enumerate(ops):
            avail = machines_by_op.get(op.NomOperation, [])
            if not avail:
                continue

            # Use the smallest machine capacity so every machine can handle any lot
            min_cap  = min(m.CapaciteMax for m in avail)
            lot_size = min(op.QuantiteLot, min_cap)
            nb_lots  = math.ceil(cmd.Quantite / lot_size)

            # Total machine occupation per lot = load + cycle + unload
            dur_total = op.DureeTotale  # TempsChargementMinutes + DureeMinutes + TempsDecharementMinutes

            for lot_idx in range(nb_lots):
                key = (cmd.NumeroCommande, op_idx, lot_idx)
                total_lots += 1

                s_var = model.NewIntVar(0, horizon - dur_total,
                    f"s_{cmd.NumeroCommande}_{op_idx}_{lot_idx}")
                e_var = model.NewIntVar(0, horizon,
                    f"e_{cmd.NumeroCommande}_{op_idx}_{lot_idx}")
                model.Add(e_var == s_var + dur_total)

                iv = model.NewIntervalVar(s_var, dur_total, e_var,
                    f"iv_{cmd.NumeroCommande}_{op_idx}_{lot_idx}")

                lot_vars[key] = {
                    "start":                    s_var,
                    "end":                      e_var,
                    "interval":                 iv,
                    "NomOperation":             op.NomOperation,
                    "DureeMinutes":             op.DureeMinutes,
                    "TempsChargementMinutes":   op.TempsChargementMinutes,
                    "TempsDecharementMinutes":  op.TempsDecharementMinutes,
                    "DureeTotale":              dur_total,
                    "LotSize":                  lot_size,
                    "NbLots":                   nb_lots,
                    "LotIdx":                   lot_idx,
                }

                # Machine assignment for this lot (optional interval per machine)
                bools = []
                for m in avail:
                    b = model.NewBoolVar(
                        f"b_{cmd.NumeroCommande}_{op_idx}_{lot_idx}_{m.Id}")
                    machine_vars[(cmd.NumeroCommande, op_idx, lot_idx, m.Id)] = b
                    bools.append(b)

                    oi = model.NewOptionalIntervalVar(
                        s_var, dur_total, e_var, b,
                        f"oi_{cmd.NumeroCommande}_{op_idx}_{lot_idx}_{m.Id}")
                    machine_itvs[m.Id].append(oi)

                model.AddExactlyOne(bools)

    # ── No overlap per machine ────────────────────────────────────────────────
    for m in machines_ok:
        if machine_itvs[m.Id]:
            model.AddNoOverlap(machine_itvs[m.Id])

    print(f"📐 {total_lots} lots-tâches individuels")

    # ── Lot chaining within same (cmd, op): lot[k+1] starts after lot[k] ends ─
    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for op_idx, op in enumerate(ops):
            avail = machines_by_op.get(op.NomOperation, [])
            if not avail:
                continue
            min_cap  = min(m.CapaciteMax for m in avail)
            lot_size = min(op.QuantiteLot, min_cap)
            nb_lots  = math.ceil(cmd.Quantite / lot_size)
            for lot_idx in range(nb_lots - 1):
                k1 = (cmd.NumeroCommande, op_idx, lot_idx)
                k2 = (cmd.NumeroCommande, op_idx, lot_idx + 1)
                if k1 in lot_vars and k2 in lot_vars:
                    model.Add(lot_vars[k2]["start"] >= lot_vars[k1]["end"])

    # ── Inter-operation precedence: last lot of op[i] before first lot of op[i+1] ─
    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for i in range(len(ops) - 1):
            op_cur  = ops[i]
            op_next = ops[i + 1]
            avail_cur  = machines_by_op.get(op_cur.NomOperation, [])
            avail_next = machines_by_op.get(op_next.NomOperation, [])
            if not avail_cur or not avail_next:
                continue

            min_cap_cur  = min(m.CapaciteMax for m in avail_cur)
            lot_size_cur = min(op_cur.QuantiteLot, min_cap_cur)
            nb_lots_cur  = math.ceil(cmd.Quantite / lot_size_cur)

            last_lot_key  = (cmd.NumeroCommande, i,   nb_lots_cur - 1)
            first_lot_key = (cmd.NumeroCommande, i+1, 0)
            if last_lot_key in lot_vars and first_lot_key in lot_vars:
                model.Add(
                    lot_vars[first_lot_key]["start"] >= lot_vars[last_lot_key]["end"]
                )

    # ── Export deadline: last lot of last op must finish in time ─────────────
    tard_vars, weights = [], []
    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        if not ops:
            continue
        last_op = ops[-1]
        avail   = machines_by_op.get(last_op.NomOperation, [])
        if not avail:
            continue
        min_cap  = min(m.CapaciteMax for m in avail)
        lot_size = min(last_op.QuantiteLot, min_cap)
        nb_lots  = math.ceil(cmd.Quantite / lot_size)
        last_key = (cmd.NumeroCommande, len(ops) - 1, nb_lots - 1)
        if last_key not in lot_vars:
            continue

        deadline = date_to_pm(cmd.DateExport)
        tard = model.NewIntVar(0, horizon, f"tard_{cmd.NumeroCommande}")
        late = model.NewIntVar(-horizon, horizon, f"late_{cmd.NumeroCommande}")
        model.Add(late == lot_vars[last_key]["end"] - deadline)
        model.AddMaxEquality(tard, [late, model.NewConstant(0)])
        tard_vars.append(tard)
        weights.append(10 // cmd.Urgence)   # lower urgence value → higher weight

    # ── Makespan ──────────────────────────────────────────────────────────────
    makespan = model.NewIntVar(0, horizon, "makespan")
    all_ends = []
    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        if not ops:
            continue
        last_op = ops[-1]
        avail   = machines_by_op.get(last_op.NomOperation, [])
        if not avail:
            continue
        min_cap  = min(m.CapaciteMax for m in avail)
        lot_size = min(last_op.QuantiteLot, min_cap)
        nb_lots  = math.ceil(cmd.Quantite / lot_size)
        last_key = (cmd.NumeroCommande, len(ops) - 1, nb_lots - 1)
        if last_key in lot_vars:
            all_ends.append(lot_vars[last_key]["end"])
    if all_ends:
        model.AddMaxEquality(makespan, all_ends)

    # ── Machine spread bonus ──────────────────────────────────────────────────
    machine_used = []
    for m in machines_ok:
        uses = [machine_vars[k] for k in machine_vars if k[3] == m.Id]
        if uses:
            mu = model.NewBoolVar(f"mu_{m.Id}")
            model.AddMaxEquality(mu, uses)
            machine_used.append(mu)

    model.Minimize(
        sum(w * t for w, t in zip(weights, tard_vars))
        + makespan
        - 100 * sum(machine_used)
    )

    # ── Solve ─────────────────────────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print(f"  RÉSOLUTION — limite {MAX_SOLVE_SECONDS}s | {NUM_WORKERS} workers")
    print(f"{'━'*70}")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAX_SOLVE_SECONDS
    solver.parameters.num_search_workers  = NUM_WORKERS
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

    # ── Extract results — one entry per lot ───────────────────────────────────
    results = []
    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        if not ops:
            continue
        print(f"📦 {cmd.NumeroCommande} | qty={cmd.Quantite} | urgence={cmd.Urgence}")

        for op_idx, op in enumerate(ops):
            avail = machines_by_op.get(op.NomOperation, [])
            if not avail:
                continue
            min_cap  = min(m.CapaciteMax for m in avail)
            lot_size = min(op.QuantiteLot, min_cap)
            nb_lots  = math.ceil(cmd.Quantite / lot_size)

            for lot_idx in range(nb_lots):
                key = (cmd.NumeroCommande, op_idx, lot_idx)
                lv  = lot_vars.get(key)
                if not lv:
                    continue

                s_pm = solver.Value(lv["start"])
                e_pm = solver.Value(lv["end"])

                s_day, s_h, s_m = pm_to_clock(s_pm)
                e_day, e_h, e_m = pm_to_clock(e_pm)
                d_start = working_day_date(s_day)
                d_end   = working_day_date(e_day)

                # Find assigned machine
                assigned_id, assigned_name = -1, "non assigné"
                for m in avail:
                    mkey = (cmd.NumeroCommande, op_idx, lot_idx, m.Id)
                    if mkey in machine_vars and solver.Value(machine_vars[mkey]):
                        assigned_id   = m.Id
                        assigned_name = f"{m.Id} ({m.NomMachine})"
                        break

                # Last lot may be a partial lot
                pieces_this_lot = (
                    cmd.Quantite - lot_idx * lot_size
                    if lot_idx == nb_lots - 1
                    else lot_size
                )

                print(f"   [{op_idx+1}.{lot_idx+1}] {op.NomOperation:20s}: "
                      f"{d_start} {s_h:02d}h{s_m:02d}→{e_h:02d}h{e_m:02d} "
                      f"[chg={op.TempsChargementMinutes}+cyc={op.DureeMinutes}+dch={op.TempsDecharementMinutes}] | "
                      f"{pieces_this_lot}pcs | {assigned_name}")

                results.append({
                    "NumeroCommande":          cmd.NumeroCommande,
                    "Quantite":                cmd.Quantite,
                    "RecetteId":               cmd.RecetteId,
                    "Urgence":                 cmd.Urgence,
                    "NomOperation":            op.NomOperation,
                    "MachineId":               assigned_id,
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
        last_op      = ops[-1]
        avail_last   = machines_by_op.get(last_op.NomOperation, [])
        if avail_last:
            min_cap_l  = min(m.CapaciteMax for m in avail_last)
            lot_size_l = min(last_op.QuantiteLot, min_cap_l)
            nb_lots_l  = math.ceil(cmd.Quantite / lot_size_l)
            last_key   = (cmd.NumeroCommande, len(ops)-1, nb_lots_l-1)
            if last_key in lot_vars:
                fin_pm  = solver.Value(lot_vars[last_key]["end"])
                fin_day = fin_pm // PPD
                if fin_day > deadline_day:
                    print(f"   ⚠️  RETARD {fin_day - deadline_day} jour(s)\n")
                else:
                    print(f"   ✅ marge {deadline_day - fin_day} jour(s)\n")

    return results


if __name__ == "__main__":
    solve()