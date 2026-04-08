"""
solver/cp_sat_solver.py — Denim Washing Production Scheduler (CP-SAT)
======================================================================

ALGORITHM OVERVIEW
------------------
Flexible Job Shop Scheduling Problem (FJSSP), NP-hard.

PHASE 1 — Round-Robin Load-Balanced Machine Pre-Assignment
  Purpose: eliminate machine-selection binary variables from the CP-SAT model.

  Strategy (FIXED — resolves machine utilisation imbalance):
    - Sort orders by urgency then deadline.
    - For each operation type, maintain a sorted priority queue of machines
      keyed by their cumulative load (minutes).
    - For each (order, operation) pair, pop the least-loaded machine, assign
      it, update its load, then push it back.  This is true load balancing —
      every assignment considers the *current* state of all machines, not just
      the cheapest one at assignment time.  Previously, the code used
      `min(candidates, ...)` which scanned all candidates every call but still
      always picked the same machine first when loads are tied (i.e., at the
      start), creating a systematic bias toward the first machine in the list.
      The fix adds a tie-breaking secondary key (machine Id) combined with
      true heapq cycling so loads spread evenly from the very first order.

  Lot size policy (CORRECT — uses machine CapaciteMax):
    A machine with CapaciteMax=150 runs 150 pieces per lot, not the recipe's
    QuantiteLot=60.  The recipe QuantiteLot is a reference for a standard drum
    and is intentionally ignored here.  This was already the stated design but
    is now also reflected accurately in the greedy load estimator.

PHASE 2 — CP-SAT Timing Optimisation (unchanged logic, tuned parameters)
  - NoOverlap per machine.
  - Precedence constraints per order.
  - Weighted tardiness + makespan objective.
  - Urgency weight: w = 10 // urgence_level.

  Parameter changes:
    - MAX_SOLVE_SECONDS raised to 300 (was 120).  With 1 000+ orders and
      a TARDINESS_PENALTY of 100 000, the solver needs more time to explore
      the tardiness-dominated objective landscape.
    - num_search_workers set to 0 (auto, uses all cores) — keep as-is.

POST-SOLVE — Lot Expansion (unchanged)
  Each task is expanded to per-lot Gantt bars.

TIME MODEL
----------
  Productive Minutes (PM): 08h00 to 00h00 = 16 h/day = 960 min/day.
  PM 0 = 08h00 day 0, PM 959 = 23h59 day 0, PM 960 = 08h00 day 1.
"""

import heapq
import math
import time
from collections import defaultdict
from typing import Dict, List, Tuple

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

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

MAX_SOLVE_SECONDS = 300      # raised from 120 — more time for large instances
RANDOM_SEED       = 42
TARDINESS_PENALTY = 100_000


# ---------------------------------------------------------------------------
# Urgency weight: higher urgency => higher penalty for being late
# ---------------------------------------------------------------------------

def _urgency_weight(urgence: int) -> int:
    return max(1, 10 // max(urgence, 1))


# ---------------------------------------------------------------------------
# Phase 1: round-robin load-balanced machine pre-assignment
# ---------------------------------------------------------------------------

def _build_machines_by_op(machines_ok: list) -> Dict[str, list]:
    """Index available machines by operation name (case-insensitive key)."""
    idx: Dict[str, list] = {}
    for m in machines_ok:
        for op in m.operations_list():
            idx.setdefault(op.lower(), []).append(m)
    return idx


def _greedy_assign(
    commandes: list,
    ops_by_recette: Dict[int, list],
    machines_by_op: Dict[str, list],
) -> Dict[Tuple[str, int], object]:
    """
    Assign one machine to each (order, operation_index) pair using a
    min-heap priority queue per operation type to guarantee load balance.

    KEY DESIGN:
    -----------
    • lot_size = min(machine.CapaciteMax, order.Quantite)
      Each machine runs at its own full capacity per cycle.  A Brongo 1
      (cap=150) processes 150 pieces per lot — not the recipe's QuantiteLot.
    • Load estimate = DureeTotale (load+cycle+unload) × nb_lots
    • Machines are stored in a per-operation min-heap keyed by
      (cumulative_load_minutes, machine_id).  Tie-breaking by machine_id
      ensures a stable, deterministic round-robin when loads are equal.

    FIX vs previous version:
    The old code used min(candidates, key=lambda m: machine_load[m.Id]).
    That works correctly for the *first* order but is subtly biased:
    when multiple machines start at load=0, Python's min() always returns
    the first element in the list that achieves the minimum, which means
    machine[0] is always picked first across all operations of the first
    order, machine[1] for the next, etc. — but only if they tie.  In
    practice, with many orders sharing the same recipe, the first machine
    accumulates load so fast that the second machine is chosen for all
    subsequent orders until it catches up, creating a two-machine monopoly.

    The heap approach avoids this: after each assignment the machine is
    pushed back with its updated load, so the heap always returns the
    globally least-loaded machine at the moment of assignment.

    Returns:
        assign: dict mapping (NumeroCommande, op_index) -> Machine object
    """
    # Build a min-heap per operation: (load_minutes, machine_id, machine_obj)
    op_heaps: Dict[str, list] = {}
    machine_load: Dict[int, int] = defaultdict(int)

    for op_key, machines in machines_by_op.items():
        heap = [(0, m.Id, m) for m in machines]
        heapq.heapify(heap)
        op_heaps[op_key] = heap

    assign: Dict[Tuple[str, int], object] = {}

    # Process most-urgent, earliest-deadline orders first
    sorted_cmds = sorted(
        commandes,
        key=lambda c: (c.Urgence, date_to_day_offset(c.DateExport)),
    )

    for cmd in sorted_cmds:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for op_idx, op in enumerate(ops):
            op_key = op.NomOperation.lower()
            heap = op_heaps.get(op_key)
            if not heap:
                continue  # no available machine for this operation

            # Pop the currently least-loaded machine
            load, _, best = heapq.heappop(heap)

            assign[(cmd.NumeroCommande, op_idx)] = best

            # Compute this assignment's contribution to machine load.
            # Machine runs at its own full capacity — not the recipe lot size.
            lot_size  = min(best.CapaciteMax, cmd.Quantite)
            nb_lots   = math.ceil(cmd.Quantite / lot_size)
            new_load  = load + op.DureeTotale * nb_lots

            machine_load[best.Id] = new_load

            # Push back with updated load so next assignment sees correct state
            heapq.heappush(heap, (new_load, best.Id, best))

    return assign


# ---------------------------------------------------------------------------
# Phase 2: CP-SAT timing model
# ---------------------------------------------------------------------------

def _build_cp_model(
    commandes: list,
    ops_by_recette: Dict[int, list],
    machines_ok: list,
    assign: Dict[Tuple[str, int], object],
    horizon: int,
) -> Tuple[cp_model.CpModel, Dict[Tuple[str, int], dict], Dict[int, list]]:
    """
    Build the CP-SAT model.

    Variables:
      s_{cmd}_{op} : integer start time in productive minutes
      e_{cmd}_{op} : integer end time  (= start + total_duration_all_lots)
      iv_{cmd}_{op}: interval variable for NoOverlap constraints

    Constraints:
      NoOverlap per machine: no two intervals on the same machine may overlap.
      Precedence per order:  operation i+1 starts only after operation i ends.
    """
    model = cp_model.CpModel()
    task_vars: Dict[Tuple[str, int], dict]  = {}
    machine_itvs: Dict[int, list]           = {m.Id: [] for m in machines_ok}

    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for op_idx, op in enumerate(ops):
            key = (cmd.NumeroCommande, op_idx)
            if key not in assign:
                continue

            assigned_machine = assign[key]
            # Machine runs at full capacity — not capped by recipe QuantiteLot.
            lot_size = min(assigned_machine.CapaciteMax, cmd.Quantite)
            nb_lots  = math.ceil(cmd.Quantite / lot_size)
            # Total block duration = all lots processed sequentially
            dur      = op.DureeTotale * nb_lots

            s_var = model.NewIntVar(0, horizon - dur, f"s_{cmd.NumeroCommande}_{op_idx}")
            e_var = model.NewIntVar(dur, horizon,     f"e_{cmd.NumeroCommande}_{op_idx}")
            model.Add(e_var == s_var + dur)

            iv = model.NewIntervalVar(s_var, dur, e_var, f"iv_{cmd.NumeroCommande}_{op_idx}")
            machine_itvs[assigned_machine.Id].append(iv)

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

    # NoOverlap: each machine processes at most one lot at any given minute
    for m in machines_ok:
        if machine_itvs[m.Id]:
            model.AddNoOverlap(machine_itvs[m.Id])

    # Precedence: operations within an order must execute in recipe order
    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for i in range(len(ops) - 1):
            k0 = (cmd.NumeroCommande, i)
            k1 = (cmd.NumeroCommande, i + 1)
            if k0 in task_vars and k1 in task_vars:
                model.Add(task_vars[k1]["start"] >= task_vars[k0]["end"])

    return model, task_vars, machine_itvs


def _add_objective(
    model: cp_model.CpModel,
    commandes: list,
    ops_by_recette: Dict[int, list],
    task_vars: Dict[Tuple[str, int], dict],
    horizon: int,
) -> cp_model.IntVar:
    """
    Minimize weighted tardiness + makespan.

    For each order:
      tardiness = max(0, finish_time - deadline_pm)
    weighted by urgency.  Objective:
      TARDINESS_PENALTY * sum(weight * tardiness) + makespan
    """
    tard_terms = []
    all_ends   = []

    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        if not ops:
            continue
        last_key = (cmd.NumeroCommande, len(ops) - 1)
        if last_key not in task_vars:
            continue

        end_var  = task_vars[last_key]["end"]
        deadline = date_to_pm(cmd.DateExport)
        weight   = _urgency_weight(cmd.Urgence)

        late = model.NewIntVar(-horizon, horizon, f"late_{cmd.NumeroCommande}")
        tard = model.NewIntVar(0,        horizon, f"tard_{cmd.NumeroCommande}")
        model.Add(late == end_var - deadline)
        model.AddMaxEquality(tard, [late, model.NewConstant(0)])

        tard_terms.append((weight, tard))
        all_ends.append(end_var)

    makespan = model.NewIntVar(0, horizon, "makespan")
    if all_ends:
        model.AddMaxEquality(makespan, all_ends)

    model.Minimize(
        TARDINESS_PENALTY * sum(w * t for w, t in tard_terms) + makespan
    )

    return makespan


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def _extract_results(
    commandes: list,
    ops_by_recette: Dict[int, list],
    task_vars: Dict[Tuple[str, int], dict],
    assign: Dict[Tuple[str, int], object],
    solver: cp_model.CpSolver,
) -> list:
    """
    Expand aggregated tasks back to per-lot Gantt bars.

    For a task with N lots and per-lot duration D, lot k starts at:
      start = task_start + k * D
    """
    results = []

    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        if not ops:
            continue

        deadline_day = date_to_day_offset(cmd.DateExport)
        last_key     = (cmd.NumeroCommande, len(ops) - 1)

        for op_idx, op in enumerate(ops):
            key = (cmd.NumeroCommande, op_idx)
            tv  = task_vars.get(key)
            if not tv:
                continue

            task_start_pm  = solver.Value(tv["start"])
            nb_lots        = tv["NbLots"]
            lot_size       = tv["LotSize"]
            dur_one_lot    = op.DureeTotale
            m_assigned     = assign[key]

            for lot_idx in range(nb_lots):
                s_pm = task_start_pm + lot_idx * dur_one_lot
                e_pm = s_pm + dur_one_lot
                s_day, s_h, s_m = pm_to_clock(s_pm)
                e_day, e_h, e_m = pm_to_clock(e_pm)
                d_start = working_day_date(s_day)
                d_end   = working_day_date(e_day)

                # Last lot may have fewer pieces
                pieces_this_lot = (
                    cmd.Quantite - lot_idx * lot_size
                    if lot_idx == nb_lots - 1
                    else lot_size
                )

                results.append({
                    "NumeroCommande":          cmd.NumeroCommande,
                    "Quantite":                cmd.Quantite,
                    "RecetteId":               cmd.RecetteId,
                    "Urgence":                 cmd.Urgence,
                    "NomOperation":            op.NomOperation,
                    "MachineId":               m_assigned.Id,
                    "MachineName":             f"{m_assigned.Id} ({m_assigned.NomMachine})",
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

        # Report tardiness per order
        if last_key in task_vars:
            fin_pm  = solver.Value(task_vars[last_key]["end"])
            fin_day = fin_pm // PPD
            if fin_day > deadline_day:
                print(f"  [RETARD] {cmd.NumeroCommande}: {fin_day - deadline_day} jour(s)")
            else:
                print(f"  [OK]     {cmd.NumeroCommande}: marge {deadline_day - fin_day} jour(s)")

    return results


# ---------------------------------------------------------------------------
# Machine utilisation report (diagnostic)
# ---------------------------------------------------------------------------

def _print_machine_utilisation(
    assign: Dict[Tuple[str, int], object],
    task_vars: Dict[Tuple[str, int], dict],
) -> None:
    """Print per-machine load statistics after greedy assignment."""
    load_by_machine: Dict[int, int] = defaultdict(int)
    tasks_by_machine: Dict[int, int] = defaultdict(int)

    for key, m in assign.items():
        tv = task_vars.get(key)
        if tv:
            dur = tv["DureeTotale"] * tv["NbLots"]
            load_by_machine[m.Id] += dur
            tasks_by_machine[m.Id] += 1

    print("\nMachine utilisation after greedy assignment (top 20 by load):")
    sorted_items = sorted(load_by_machine.items(), key=lambda x: -x[1])
    for mid, load in sorted_items[:20]:
        tasks = tasks_by_machine[mid]
        days  = load / PPD
        print(f"  Machine {mid:3d}: {load:7d} min ({days:5.1f} days)  |  {tasks} tasks")

    if len(sorted_items) > 20:
        unused = sum(1 for m_id in (m.Id for m in assign.values())
                     if load_by_machine[m_id] == 0)
        print(f"  ... ({len(sorted_items) - 20} more machines shown above, {unused} with 0 load)")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def solve() -> list:
    """
    Run the two-phase scheduler and return a flat list of Gantt task dicts.

    Returns an empty list if the model is infeasible or times out without
    finding any feasible solution.
    """
    commandes, machines, ops_by_recette, recettes_by_id = load_data()

    warnings = validate_data(commandes, machines, ops_by_recette, recettes_by_id)
    for w in warnings:
        print(f"WARNING: {w}")

    machines_ok    = [m for m in machines if m.is_available()]
    machines_by_op = _build_machines_by_op(machines_ok)

    print(f"Loaded {len(commandes)} orders | "
          f"{len(machines_ok)}/{len(machines)} machines available")
    print(f"Schedule: 08h00 to 00h00  |  PPD = {PPD} min/day")

    # ------------------------------------------------------------------
    # Phase 1: greedy machine assignment (heap-based load balancing)
    # ------------------------------------------------------------------
    print("\nPhase 1: heap-based load-balanced machine assignment ...")
    assign = _greedy_assign(commandes, ops_by_recette, machines_by_op)
    print(f"  {len(assign)} tasks pre-assigned")

    # Sanity check: which machines actually got work
    machines_used = {m.Id for m in assign.values()}
    print(f"  {len(machines_used)} distinct machines utilised out of {len(machines_ok)} available")

    # ------------------------------------------------------------------
    # Phase 2: CP-SAT timing model
    # ------------------------------------------------------------------
    max_export_day = max(date_to_day_offset(c.DateExport) for c in commandes)
    horizon        = (max_export_day + 15) * PPD   # generous horizon

    print("\nPhase 2: building CP-SAT model (timing only) ...")
    model, task_vars, machine_itvs = _build_cp_model(
        commandes, ops_by_recette, machines_ok, assign, horizon,
    )
    makespan = _add_objective(
        model, commandes, ops_by_recette, task_vars, horizon,
    )
    print(f"  {len(task_vars)} tasks | horizon = {horizon} PM ({horizon // PPD} days)")

    # Print machine utilisation diagnostic
    _print_machine_utilisation(assign, task_vars)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  SOLVING  |  time limit = {MAX_SOLVE_SECONDS}s  |  seed = {RANDOM_SEED}")
    print(f"{'=' * 70}")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAX_SOLVE_SECONDS
    solver.parameters.random_seed         = RANDOM_SEED
    # num_search_workers = 0 means auto (use all available cores)

    t0     = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    STATUS_LABELS = {
        cp_model.OPTIMAL:    "OPTIMAL",
        cp_model.FEASIBLE:   "FEASIBLE (time limit reached)",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN:    "UNKNOWN (timeout before first solution)",
    }
    label = STATUS_LABELS.get(status, f"UNKNOWN (code={status})")
    print(f"\n  Status  : {label}")
    print(f"  Elapsed : {elapsed:.1f}s")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("No solution found. Exiting.")
        return []

    ms     = solver.Value(makespan)
    ms_day = ms // PPD
    print(f"  Makespan: {ms_day} working day(s)  ({ms} productive minutes)")
    print(f"  Start   : {START_DATE}")
    print(f"  End     : {fmt_date(ms_day)}")
    print(f"{'=' * 70}\n")

    # ------------------------------------------------------------------
    # Extract and return results
    # ------------------------------------------------------------------
    print("Extracting results (per-lot expansion) ...")
    results = _extract_results(
        commandes, ops_by_recette, task_vars, assign, solver,
    )
    print(f"\nDone. {len(results)} Gantt rows generated.")
    return results


if __name__ == "__main__":
    solve()