"""
solver/cp_sat_solver.py — Denim Washing Production Scheduler
=============================================================

HYBRID ARCHITECTURE
-------------------
LAYER 1 — COUCHE HEURISTIQUE (Large Neighborhood Search)
LAYER 2 — COUCHE CP-SAT (timing optimisation only)

LOT SIZE RULE
-------------
    lot_size = min(op.QuantiteLot, machine.CapaciteMax, cmd.Quantite)

    The recipe defines the target lot size (op.QuantiteLot).
    The machine capacity is a hard physical ceiling (machine.CapaciteMax).
    The command quantity is the upper bound (cmd.Quantite).
    The smallest of the three wins.
"""

import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from utils.data_loader import load_data, validate_data
from utils.time_utils import (
    PPD,
    START_DATE,
    date_to_day_offset,
    date_to_pm,
    fmt_date,
    pm_to_clock,
    working_day_date,
)

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

LNS_ITERATIONS   = 300
LNS_TIME_LIMIT   = 45.0
DESTROY_RATIO    = 0.25
RANDOM_SEED      = 42
MAX_SOLVE_SECONDS = 90
TARDINESS_PENALTY = 100_000


def _urgency_weight(urgence: int) -> int:
    return max(1, 10 // max(urgence, 1))


def _priority_score(cmd, ops_by_recette: dict) -> float:
    ops = ops_by_recette.get(cmd.RecetteId, [])
    total_duration = sum(op.DureeTotale for op in ops)

    deadline_pm = date_to_pm(cmd.DateExport)

    # Higher score = higher priority (so we invert deadline)
    return (
        cmd.Urgence * 1000        # strong priority
        - deadline_pm             # earlier deadline = higher priority
        + total_duration          # longer jobs get slight priority
    )


def _effective_lot_size(cmd, op, machine) -> int:
    """
    Effective lot size = min(recipe lot, machine capacity, order quantity).
    The recipe is the reference; the machine capacity is the physical ceiling.
    """
    return min(op.QuantiteLot, machine.CapaciteMax, cmd.Quantite)


# ===========================================================================
# LAYER 1 — COUCHE HEURISTIQUE (Large Neighborhood Search)
# ===========================================================================

class LNSState:
    def __init__(self, commandes, ops_by_recette, machines_by_op):
        self.commandes      = commandes
        self.ops_by_recette = ops_by_recette
        self.machines_by_op = machines_by_op

        self.assign:       Dict[Tuple[str, int], object] = {}
        self.machine_load: Dict[int, int]                = defaultdict(int)
        self.task_load:    Dict[Tuple[str, int], int]    = {}
        self.lot_size:     Dict[Tuple[str, int], int]    = {}
        self.nb_lots:      Dict[Tuple[str, int], int]    = {}

    # -----------------------------------------------------------------------
    def _compute_task_load(self, cmd, op, machine) -> int:
        ls = _effective_lot_size(cmd, op, machine)
        nb = math.ceil(cmd.Quantite / ls)
        return op.DureeTotale * nb

    def _best_machine_for(self, cmd, op):
        op_key     = op.NomOperation.lower()
        candidates = self.machines_by_op.get(op_key, [])
        if not candidates:
            return None

        # Choose machine with lowest projected load
        return min(
            candidates,
            key=lambda m: self.machine_load[m.Id] + self._compute_task_load(cmd, op, m)
        )

    # -----------------------------------------------------------------------
    def assign_task(self, cmd, op_idx: int, op, machine) -> None:
        key = (cmd.NumeroCommande, op_idx)
        if key in self.assign:
            old_m    = self.assign[key]
            old_load = self.task_load.get(key, 0)
            self.machine_load[old_m.Id] -= old_load

        load = self._compute_task_load(cmd, op, machine)
        self.assign[key]    = machine
        self.task_load[key] = load
        self.machine_load[machine.Id] += load

        ls = _effective_lot_size(cmd, op, machine)
        self.lot_size[key] = ls
        self.nb_lots[key]  = math.ceil(cmd.Quantite / ls)

    def unassign_task(self, cmd_num: str, op_idx: int) -> Optional[object]:
        key = (cmd_num, op_idx)
        if key not in self.assign:
            return None
        machine = self.assign.pop(key)
        load    = self.task_load.pop(key, 0)
        self.machine_load[machine.Id] -= load
        self.lot_size.pop(key, None)
        self.nb_lots.pop(key, None)
        return machine

    # -----------------------------------------------------------------------
    def score(self) -> float:
        score = 0.0
        for cmd in self.commandes:
            ops = self.ops_by_recette.get(cmd.RecetteId, [])
            if not ops:
                continue
            total_load = sum(
                self.task_load.get((cmd.NumeroCommande, i), 0)
                for i in range(len(ops))
            )
            deadline_pm    = date_to_pm(cmd.DateExport)
            weight         = _urgency_weight(cmd.Urgence)
            estimated_late = max(0, total_load - deadline_pm)
            score += weight * estimated_late

        loads = [v for v in self.machine_load.values() if v > 0]
        if len(loads) > 1:
            avg      = sum(loads) / len(loads)
            variance = sum((l - avg) ** 2 for l in loads) / len(loads)
            score   += math.sqrt(variance) * 0.01
        return score

    def task_score(self, cmd_num: str, op_idx: int) -> float:
        key = (cmd_num, op_idx)
        if key not in self.assign:
            return 0.0
        machine = self.assign[key]
        return float(self.machine_load.get(machine.Id, 0))

    def copy_assign(self) -> Tuple[dict, dict, dict, dict]:
        return (
            dict(self.assign),
            dict(self.machine_load),
            dict(self.task_load),
            dict(self.lot_size),
        )

    def restore_assign(self, snapshot: Tuple) -> None:
        assign, machine_load, task_load, lot_size = snapshot
        self.assign       = dict(assign)
        self.machine_load = defaultdict(int, machine_load)
        self.task_load    = dict(task_load)
        self.lot_size     = dict(lot_size)
        for key, ls in self.lot_size.items():
            cmd_num, op_idx = key
            cmd = next((c for c in self.commandes if c.NumeroCommande == cmd_num), None)
            if cmd:
                self.nb_lots[key] = math.ceil(cmd.Quantite / ls)


def _build_machines_by_op(machines_ok: list) -> Dict[str, list]:
    idx: Dict[str, list] = {}
    for m in machines_ok:
        for op in m.operations_list():
            idx.setdefault(op.lower(), []).append(m)
    return idx


def _construct_initial_solution(state: LNSState) -> None:
    sorted_cmds = sorted(
        state.commandes,
        key=lambda c: _priority_score(c, state.ops_by_recette),
    )
    for cmd in sorted_cmds:
        ops = state.ops_by_recette.get(cmd.RecetteId, [])
        for op_idx, op in enumerate(ops):
            machine = state._best_machine_for(cmd, op)
            if machine is None:
                continue
            state.assign_task(cmd, op_idx, op, machine)


def _lns_destroy(state: LNSState, rng: random.Random) -> List[Tuple[str, int]]:
    all_keys  = list(state.assign.keys())
    n_destroy = max(1, int(len(all_keys) * DESTROY_RATIO))
    scored    = [(state.task_score(k[0], k[1]), k) for k in all_keys]
    scored.sort(key=lambda x: -x[0])

    n_score  = max(1, int(n_destroy * 0.7))
    n_random = n_destroy - n_score

    to_destroy = [k for _, k in scored[:n_score]]
    pool       = [k for _, k in scored[n_score:]]
    if pool and n_random > 0:
        to_destroy.extend(rng.sample(pool, min(n_random, len(pool))))
    return to_destroy


def _lns_repair(state: LNSState, destroyed_keys: List[Tuple[str, int]]) -> None:
    cmd_lookup      = {c.NumeroCommande: c for c in state.commandes}
    tasks_to_repair = []

    for key in destroyed_keys:
        cmd_num, op_idx = key
        state.unassign_task(cmd_num, op_idx)
        cmd = cmd_lookup.get(cmd_num)
        if cmd is None:
            continue
        ops = state.ops_by_recette.get(cmd.RecetteId, [])
        if op_idx >= len(ops):
            continue
        tasks_to_repair.append((cmd, op_idx, ops[op_idx]))

    tasks_to_repair.sort(key=lambda t: _priority_score(t[0], state.ops_by_recette))
    for cmd, op_idx, op in tasks_to_repair:
        machine = state._best_machine_for(cmd, op)
        if machine is None:
            continue
        state.assign_task(cmd, op_idx, op, machine)


def run_lns(commandes, ops_by_recette, machines_ok) -> LNSState:
    rng            = random.Random(RANDOM_SEED)
    machines_by_op = _build_machines_by_op(machines_ok)
    state          = LNSState(commandes, ops_by_recette, machines_by_op)

    print("  [LNS] Constructing initial solution ...")
    _construct_initial_solution(state)
    best_score    = state.score()
    best_snapshot = state.copy_assign()

    print(f"  [LNS] Initial score = {best_score:,.1f}  |  {len(state.assign)} tasks assigned")
    print(f"  [LNS] Running {LNS_ITERATIONS} iterations (time limit = {LNS_TIME_LIMIT}s) ...")

    t_start = time.time()
    improvements = 0

    for iteration in range(LNS_ITERATIONS):
        if time.time() - t_start > LNS_TIME_LIMIT:
            print(f"  [LNS] Time limit reached at iteration {iteration}.")
            break

        snapshot  = state.copy_assign()
        destroyed = _lns_destroy(state, rng)
        _lns_repair(state, destroyed)
        new_score = state.score()

        if new_score < best_score:
            best_score    = new_score
            best_snapshot = state.copy_assign()
            improvements += 1
        else:
            state.restore_assign(snapshot)

    elapsed = time.time() - t_start
    print(f"  [LNS] Finished: {improvements} improvements in {elapsed:.1f}s  |  best score = {best_score:,.1f}")
    state.restore_assign(best_snapshot)
    return state


# ===========================================================================
# LAYER 2 — COUCHE CP-SAT (timing optimisation)
# ===========================================================================

def _build_cp_model(commandes, ops_by_recette, machines_ok, lns_state, horizon):
    model         = cp_model.CpModel()
    task_vars     = {}
    machine_itvs  = {m.Id: [] for m in machines_ok}

    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for op_idx, op in enumerate(ops):
            key = (cmd.NumeroCommande, op_idx)
            if key not in lns_state.assign:
                continue

            machine  = lns_state.assign[key]
            lot_size = lns_state.lot_size[key]
            nb_lots  = lns_state.nb_lots[key]
            dur      = op.DureeTotale * nb_lots

            s_var = model.NewIntVar(0, horizon - dur, f"s_{cmd.NumeroCommande}_{op_idx}")
            e_var = model.NewIntVar(dur, horizon,     f"e_{cmd.NumeroCommande}_{op_idx}")
            model.Add(e_var == s_var + dur)

            iv = model.NewIntervalVar(s_var, dur, e_var, f"iv_{cmd.NumeroCommande}_{op_idx}")
            machine_itvs[machine.Id].append(iv)

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

    for m in machines_ok:
        if machine_itvs[m.Id]:
            model.AddNoOverlap(machine_itvs[m.Id])

    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for i in range(len(ops) - 1):
            k0 = (cmd.NumeroCommande, i)
            k1 = (cmd.NumeroCommande, i + 1)
            if k0 in task_vars and k1 in task_vars:
                model.Add(task_vars[k1]["start"] >= task_vars[k0]["end"])

    return model, task_vars, machine_itvs


def _add_objective(model, commandes, ops_by_recette, task_vars, horizon):
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

    model.Minimize(TARDINESS_PENALTY * sum(w * t for w, t in tard_terms) + makespan)
    return makespan


def _extract_results(commandes, ops_by_recette, task_vars, lns_state, solver):
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

            task_start_pm = solver.Value(tv["start"])
            nb_lots       = tv["NbLots"]
            lot_size      = tv["LotSize"]          # effective lot size (recipe ∩ machine cap)
            dur_one_lot   = op.DureeTotale
            machine       = lns_state.assign[key]

            for lot_idx in range(nb_lots):
                s_pm = task_start_pm + lot_idx * dur_one_lot
                e_pm = s_pm + dur_one_lot
                s_day, s_h, s_m = pm_to_clock(s_pm)
                e_day, e_h, e_m = pm_to_clock(e_pm)
                d_start = working_day_date(s_day)
                d_end   = working_day_date(e_day)

                # Last lot may be smaller than lot_size
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
                    "MachineId":               machine.Id,
                    "MachineName":             f"{machine.Id} ({machine.NomMachine})",
                    "StartPM":                 s_pm,
                    "EndPM":                   e_pm,
                    "DureeMinutes":            op.DureeMinutes,
                    "TempsChargementMinutes":  op.TempsChargementMinutes,
                    "TempsDecharementMinutes": op.TempsDecharementMinutes,
                    "DureeTotale":             op.DureeTotale,
                    "NbCycles":                1,
                    "LotSize":                 pieces_this_lot,  # pcs in THIS lot
                    "QuantiteLot":             lot_size,         # target lot size
                    "LotIdx":                  lot_idx,
                    "NbLots":                  nb_lots,
                    "DateStart":               d_start.isoformat(),
                    "DateEnd":                 d_end.isoformat(),
                    "DateExport":              cmd.DateExport,
                })

        if last_key in task_vars:
            fin_pm       = solver.Value(task_vars[last_key]["end"])
            deadline_pm  = date_to_pm(cmd.DateExport)   # = (deadline_day + 1) * PPD
            if fin_pm > deadline_pm:
                fin_day  = fin_pm // PPD
                print(f"  [RETARD] {cmd.NumeroCommande}: fini jour {fin_day} > deadline jour {deadline_day}")
            else:
                slack_days = (deadline_pm - fin_pm) // PPD
                print(f"  [OK]     {cmd.NumeroCommande}: marge ~{slack_days} jour(s)")

    return results


def _print_machine_utilisation(lns_state: LNSState) -> None:
    load_by_machine  = defaultdict(int)
    tasks_by_machine = defaultdict(int)

    for key, machine in lns_state.assign.items():
        load = lns_state.task_load.get(key, 0)
        load_by_machine[machine.Id]  += load
        tasks_by_machine[machine.Id] += 1

    print("\nMachine utilisation after LNS assignment (top 20 by load):")
    sorted_items = sorted(load_by_machine.items(), key=lambda x: -x[1])
    for mid, load in sorted_items[:20]:
        tasks = tasks_by_machine[mid]
        days  = load / PPD
        print(f"  Machine {mid:3d}: {load:7d} min ({days:5.1f} days)  |  {tasks} tasks")
    if len(sorted_items) > 20:
        print(f"  ... ({len(sorted_items) - 20} more machines)")


# ===========================================================================
# Public entry point
# ===========================================================================

def solve() -> list:
    commandes, machines, ops_by_recette, recettes_by_id = load_data()

    warnings = validate_data(commandes, machines, ops_by_recette, recettes_by_id)
    for w in warnings:
        print(f"WARNING: {w}")

    machines_ok = [m for m in machines if m.is_available()]

    print(f"Loaded {len(commandes)} orders | {len(machines_ok)}/{len(machines)} machines available")
    print(f"Schedule: 00h00 to 00h00  |  PPD = {PPD} min/day")

    separator = "=" * 70
    print(f"\n{separator}")
    print("  LAYER 1 — COUCHE HEURISTIQUE  (Large Neighborhood Search)")
    print(f"{separator}")

    lns_state = run_lns(commandes, ops_by_recette, machines_ok)

    print(f"  {len(lns_state.assign)} tasks assigned across "
          f"{len({m.Id for m in lns_state.assign.values()})} machines")
    _print_machine_utilisation(lns_state)

    max_export_day = max(date_to_day_offset(c.DateExport) for c in commandes)
    horizon        = (max_export_day + 15) * PPD

    print(f"\n{separator}")
    print("  LAYER 2 — COUCHE CP-SAT  (timing optimisation)")
    print(f"{separator}")
    print(f"  Building CP-SAT model  |  horizon = {horizon} PM ({horizon // PPD} days) ...")

    model, task_vars, machine_itvs = _build_cp_model(
        commandes, ops_by_recette, machines_ok, lns_state, horizon,
    )
    makespan = _add_objective(model, commandes, ops_by_recette, task_vars, horizon)
    print(f"  {len(task_vars)} task variables  |  "
          f"{sum(len(v) for v in machine_itvs.values())} interval constraints")

    print(f"\n{separator}")
    print(f"  SOLVING  |  time limit = {MAX_SOLVE_SECONDS}s  |  seed = {RANDOM_SEED}")
    print(f"{separator}")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAX_SOLVE_SECONDS
    solver.parameters.random_seed         = RANDOM_SEED
    solver.parameters.num_search_workers  = 8
    solver.parameters.log_search_progress = False

    t0      = time.time()
    status  = solver.Solve(model)
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
    ms_rem = ms % PPD
    if ms_day == 0:
        ms_h   = ms_rem // 60
        ms_min = ms_rem % 60
        ms_str = f"{ms_h}h{ms_min:02d}min  ({ms} productive minutes)"
    else:
        ms_str = f"{ms_day} working day(s)  ({ms} productive minutes)"
    print(f"  Makespan: {ms_str}")
    print(f"  Start   : {START_DATE}")
    print(f"  End     : {fmt_date(ms_day)}")
    print(f"{separator}\n")

    print("Extracting results (per-lot expansion) ...")
    results = _extract_results(commandes, ops_by_recette, task_vars, lns_state, solver)
    print(f"\nDone. {len(results)} Gantt rows generated.")
    return results


if __name__ == "__main__":
    solve()