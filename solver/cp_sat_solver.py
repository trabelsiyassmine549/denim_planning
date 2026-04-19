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
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False

from models.domain import Commande, Machine, OperationRecette
from models.schemas import GanttRow
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


def _lns_destroy(state: LNSState) -> List[Tuple[str, int]]:
    """
    Destroy strategy: select the top-N tasks by machine load (highest-loaded
    machine first). This matches the original monolithic api.py exactly —
    no randomness injected, which keeps the LNS deterministic and reproducible.
    """
    all_keys  = list(state.assign.keys())
    n_destroy = max(1, int(len(all_keys) * DESTROY_RATIO))

    # Score each task by the total load on its assigned machine
    scored = sorted(
        all_keys,
        key=lambda k: -state.machine_load.get(
            state.assign[k].Id, 0
        ),
    )
    return scored[:n_destroy]


def _lns_repair(state: LNSState, destroyed_keys: List[Tuple[str, int]]) -> None:
    """
    Repair strategy: unassign all destroyed tasks first, then re-assign them
    in the same order they were destroyed (no re-sort). This matches the
    original monolithic api.py exactly.
    """
    cmd_lookup = {c.NumeroCommande: c for c in state.commandes}

    # Collect tasks to repair (unassign as we go)
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

    # Re-assign in original order (no priority re-sort — matches original)
    for cmd, op_idx, op in tasks_to_repair:
        machine = state._best_machine_for(cmd, op)
        if machine is None:
            continue
        state.assign_task(cmd, op_idx, op, machine)


def run_lns(commandes, ops_by_recette, machines_ok) -> LNSState:
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
        destroyed = _lns_destroy(state)
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


def _extract_results(commandes, ops_by_recette, task_vars, lns_state, solver) -> List[GanttRow]:
    rows: List[GanttRow] = []

    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        if not ops:
            continue

        for op_idx, op in enumerate(ops):
            key = (cmd.NumeroCommande, op_idx)
            tv  = task_vars.get(key)
            if not tv:
                continue

            task_start_pm = solver.Value(tv["start"])
            nb_lots       = tv["NbLots"]
            lot_size      = tv["LotSize"]
            machine       = lns_state.assign[key]

            for lot_idx in range(nb_lots):
                s_pm  = task_start_pm + lot_idx * op.DureeTotale
                e_pm  = s_pm + op.DureeTotale
                s_day = pm_to_clock(s_pm)[0]
                e_day = pm_to_clock(e_pm)[0]

                pieces = (
                    cmd.Quantite - lot_idx * lot_size
                    if lot_idx == nb_lots - 1
                    else lot_size
                )

                rows.append(GanttRow(
                    numeroCommande          = cmd.NumeroCommande,
                    quantite                = cmd.Quantite,
                    recetteId               = cmd.RecetteId,
                    urgence                 = cmd.Urgence,
                    nomOperation            = op.NomOperation,
                    machineId               = machine.Id,
                    machineName             = machine.NomMachine,
                    startPM                 = s_pm,
                    endPM                   = e_pm,
                    dureeMinutes            = op.DureeMinutes,
                    tempsChargementMinutes  = op.TempsChargementMinutes,
                    tempsDecharementMinutes = op.TempsDecharementMinutes,
                    dureeTotale             = op.DureeTotale,
                    lotSize                 = pieces,
                    quantiteLot             = lot_size,
                    lotIdx                  = lot_idx,
                    nbLots                  = nb_lots,
                    dateStart               = working_day_date(s_day).isoformat(),
                    dateEnd                 = working_day_date(e_day).isoformat(),
                    dateExport              = cmd.DateExport,
                ))

    return rows


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
# Public API — used by api.py
# ===========================================================================

class SolverResult:
    """Thin wrapper returned by run_cpsat so api.py stays decoupled from cp_model internals."""

    def __init__(self, solver, task_vars, makespan_pm: int):
        self._solver     = solver
        self._task_vars  = task_vars
        self.makespan_pm = makespan_pm

    def extract_rows(self, commandes, ops_by_recette, lns_state) -> List[GanttRow]:
        return _extract_results(commandes, ops_by_recette, self._task_vars, lns_state, self._solver)


def run_cpsat(commandes, ops_by_recette, machines_ok, lns_state, horizon) -> SolverResult:
    """
    Build and solve the CP-SAT model.
    Returns a SolverResult with makespan_pm and an extract_rows() method.
    Raises RuntimeError if OR-Tools is unavailable or no solution is found.
    """
    if not HAS_ORTOOLS:
        raise RuntimeError("OR-Tools is not installed")

    model, task_vars, _ = _build_cp_model(
        commandes, ops_by_recette, machines_ok, lns_state, horizon
    )
    makespan_var = _add_objective(model, commandes, ops_by_recette, task_vars, horizon)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAX_SOLVE_SECONDS
    solver.parameters.random_seed         = RANDOM_SEED
    solver.parameters.num_search_workers  = 4
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT found no feasible solution")

    return SolverResult(solver, task_vars, solver.Value(makespan_var))


def lns_fallback(commandes, ops_by_recette, lns_state) -> List[GanttRow]:
    """
    Greedy sequential schedule used when OR-Tools is unavailable.
    Respects operation precedence and machine non-overlap (sequentially).
    """
    machine_end: Dict[int, int] = defaultdict(int)
    rows: List[GanttRow] = []

    for cmd in sorted(commandes, key=lambda c: c.Urgence):
        ops      = ops_by_recette.get(cmd.RecetteId, [])
        prev_end = 0

        for op_idx, op in enumerate(ops):
            key     = (cmd.NumeroCommande, op_idx)
            machine = lns_state.assign.get(key)
            if machine is None:
                continue

            nb_lots  = lns_state.nb_lots.get(key, 1)
            lot_size = lns_state.lot_size.get(key, op.QuantiteLot)
            s_pm     = max(machine_end[machine.Id], prev_end)
            e_pm     = s_pm + op.DureeTotale * nb_lots

            machine_end[machine.Id] = e_pm
            prev_end                = e_pm

            for lot_idx in range(nb_lots):
                sp    = s_pm + lot_idx * op.DureeTotale
                ep    = sp + op.DureeTotale
                s_day = pm_to_clock(sp)[0]
                e_day = pm_to_clock(ep)[0]
                pieces = (
                    cmd.Quantite - lot_idx * lot_size
                    if lot_idx == nb_lots - 1
                    else lot_size
                )
                rows.append(GanttRow(
                    numeroCommande          = cmd.NumeroCommande,
                    quantite                = cmd.Quantite,
                    recetteId               = cmd.RecetteId,
                    urgence                 = cmd.Urgence,
                    nomOperation            = op.NomOperation,
                    machineId               = machine.Id,
                    machineName             = machine.NomMachine,
                    startPM                 = sp,
                    endPM                   = ep,
                    dureeMinutes            = op.DureeMinutes,
                    tempsChargementMinutes  = op.TempsChargementMinutes,
                    tempsDecharementMinutes = op.TempsDecharementMinutes,
                    dureeTotale             = op.DureeTotale,
                    lotSize                 = pieces,
                    quantiteLot             = lot_size,
                    lotIdx                  = lot_idx,
                    nbLots                  = nb_lots,
                    dateStart               = working_day_date(s_day).isoformat(),
                    dateEnd                 = working_day_date(e_day).isoformat(),
                    dateExport              = cmd.DateExport,
                ))

    return rows


# ===========================================================================
# Public entry point (standalone CLI)
# ===========================================================================
 
def solve() -> List[GanttRow]:
    """
    Full solve pipeline for CLI / standalone use.
    Loads data via utils.data_loader, runs LNS → CP-SAT, returns Gantt rows.
    """
    from utils.data_loader import load_data, validate_data   # type: ignore

    commandes, machines, ops_by_recette, recettes_by_id = load_data()

    for w in validate_data(commandes, machines, ops_by_recette, recettes_by_id):
        print(f"WARNING: {w}")

    machines_ok = [m for m in machines if m.is_available()]
    print(
        f"Loaded {len(commandes)} orders | "
        f"{len(machines_ok)}/{len(machines)} machines available"
    )
    print(f"Schedule: 00h00 to 00h00  |  PPD = {PPD} min/day")

    sep = "=" * 70
    print(f"\n{sep}\n  LAYER 1 — COUCHE HEURISTIQUE  (Large Neighborhood Search)\n{sep}")

    lns_state = run_lns(commandes, ops_by_recette, machines_ok)
    print(
        f"  {len(lns_state.assign)} tasks assigned across "
        f"{len({m.Id for m in lns_state.assign.values()})} machines"
    )
    _print_machine_utilisation(lns_state)

    max_export_day = max(date_to_day_offset(c.DateExport) for c in commandes)
    horizon        = (max_export_day + 15) * PPD

    print(f"\n{sep}\n  LAYER 2 — COUCHE CP-SAT  (timing optimisation)\n{sep}")
    print(f"  Building CP-SAT model  |  horizon = {horizon} PM ({horizon // PPD} days) ...")

    result = run_cpsat(commandes, ops_by_recette, machines_ok, lns_state, horizon)

    ms        = result.makespan_pm
    ms_day    = ms // PPD
    ms_rem    = ms % PPD
    ms_h, ms_min = ms_rem // 60, ms_rem % 60
    ms_str    = (
        f"{ms_h}h{ms_min:02d}min  ({ms} productive minutes)"
        if ms_day == 0
        else f"{ms_day} working day(s)  ({ms} productive minutes)"
    )
    print(f"  Makespan: {ms_str}")
    print(f"  Start   : {START_DATE}")
    print(f"  End     : {fmt_date(ms_day)}")
    print(f"{sep}\n")

    print("Extracting results (per-lot expansion) ...")
    rows = result.extract_rows(commandes, ops_by_recette, lns_state)
    print(f"\nDone. {len(rows)} Gantt rows generated.")
    return rows
 
 
if __name__ == "__main__":
    solve()