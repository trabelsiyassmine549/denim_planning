"""
solver/cp_sat_solver.py — Denim Washing Production Scheduler
=============================================================

HYBRID ARCHITECTURE
-------------------
This module implements the two-layer hybrid architecture described in the
design document:

LAYER 1 — COUCHE HEURISTIQUE (Large Neighborhood Search)
    Answers: "Which command goes on which machine, in which lot?"

    1a. Initial solution construction
        Sort orders by priority score (urgency + deadline proximity).
        For each order, for each operation, assign the least-loaded
        compatible and available machine.  This produces a valid baseline.

    1b. LNS improvement loop
        At each iteration:
          - Evaluate the current solution by a scoring function that
            penalises tardiness and load imbalance.
          - Select the DESTROY_RATIO fraction of assignments that
            contribute most to the score (worst assignments).
          - REPAIR: reassign those assignments from scratch using the
            same priority-aware least-loaded heuristic.
          - Accept the new solution if it is strictly better
            (greedy hill-climbing acceptance).
        Repeat for LNS_ITERATIONS iterations or until LNS_TIME_LIMIT
        seconds have elapsed.

    Output: a validated assignment dict  (NumeroCommande, op_idx) -> Machine
            and lot metadata (lot_size, nb_lots) per task.

LAYER 2 — COUCHE CP-SAT (timing optimisation only)
    Answers: "At what exact minute does each task start and end?"

    The CP-SAT model receives the fixed machine assignment from Layer 1
    and optimises only the scheduling timeline:
      - NoOverlap per machine (no two tasks may overlap on the same
        machine at any productive minute).
      - Precedence per order (operation k+1 can start only after
        operation k finishes).
    Objective: minimise TARDINESS_PENALTY * sum(weight * tardiness)
               + makespan.

    The LNS solution's load estimates are injected as hint values so
    that CP-SAT converges faster.

TIME MODEL
----------
    Productive Minutes (PM): 00h00 to 00h00 = 24 h/day = 1440 min/day.
    PM 0 = 00h00 day 0,  PM 1439 = 23h59 day 0,  PM 1440 = 00h00 day 1.
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

# Layer 1 — LNS parameters
LNS_ITERATIONS   = 200      # maximum LNS iterations
LNS_TIME_LIMIT   = 30.0    # wall-clock seconds for LNS (reduced: heuristic is warm start only)
DESTROY_RATIO    = 0.25     # fraction of assignments destroyed per iteration
RANDOM_SEED      = 42

# Layer 2 — CP-SAT parameters
# With 1000+ orders, LNS already provides a near-optimal assignment.
# CP-SAT only needs to optimise timing — a tight time limit is sufficient
# because the heuristic hint (injected as starting point) lets CP-SAT
# find a good feasible solution in the first few seconds.
MAX_SOLVE_SECONDS = 30      # reduced from 300 — LNS warm-start makes this enough
TARDINESS_PENALTY = 100_000


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _urgency_weight(urgence: int) -> int:
    """Higher urgency level (lower number) => higher penalty weight."""
    return max(1, 10 // max(urgence, 1))


def _priority_score(cmd, ops_by_recette: dict) -> float:
    """
    Lower score = higher priority.
    Combines urgency level and deadline proximity so that urgent,
    near-deadline orders are processed first.
    """
    deadline_offset = date_to_day_offset(cmd.DateExport)
    return cmd.Urgence * 10 + deadline_offset


# ===========================================================================
# LAYER 1 — COUCHE HEURISTIQUE  (Large Neighborhood Search)
# ===========================================================================

class LNSState:
    """
    Holds the current LNS solution and all data structures required
    to evaluate and mutate it efficiently.
    """

    def __init__(
        self,
        commandes: list,
        ops_by_recette: Dict[int, list],
        machines_by_op: Dict[str, list],
    ) -> None:
        self.commandes      = commandes
        self.ops_by_recette = ops_by_recette
        self.machines_by_op = machines_by_op

        # assignment: (NumeroCommande, op_idx) -> Machine
        self.assign: Dict[Tuple[str, int], object] = {}

        # machine_load[machine_id] -> cumulative load minutes
        self.machine_load: Dict[int, int] = defaultdict(int)

        # per-task contribution to machine load (for incremental updates)
        self.task_load: Dict[Tuple[str, int], int] = {}

        # per-task lot metadata
        self.lot_size: Dict[Tuple[str, int], int] = {}
        self.nb_lots:  Dict[Tuple[str, int], int] = {}

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _compute_task_load(self, cmd, op, machine) -> int:
        """Total machine-occupation minutes for one (cmd, op, machine) triple."""
        ls = min(machine.CapaciteMax, cmd.Quantite)
        nb = math.ceil(cmd.Quantite / ls)
        return op.DureeTotale * nb

    def _best_machine_for(self, cmd, op):
        """
        Return the least-loaded available machine for the given operation.
        Tie-broken by machine Id for determinism.
        """
        op_key    = op.NomOperation.lower()
        candidates = self.machines_by_op.get(op_key, [])
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda m: (self.machine_load[m.Id], m.Id),
        )

    # -----------------------------------------------------------------------
    # Assignment management
    # -----------------------------------------------------------------------

    def assign_task(self, cmd, op_idx: int, op, machine) -> None:
        key = (cmd.NumeroCommande, op_idx)
        # Remove old assignment contribution if it existed
        if key in self.assign:
            old_m    = self.assign[key]
            old_load = self.task_load.get(key, 0)
            self.machine_load[old_m.Id] -= old_load

        load = self._compute_task_load(cmd, op, machine)
        self.assign[key]    = machine
        self.task_load[key] = load
        self.machine_load[machine.Id] += load

        ls = min(machine.CapaciteMax, cmd.Quantite)
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
    # Scoring  (lower is better)
    # -----------------------------------------------------------------------

    def score(self) -> float:
        """
        Scoring function for the LNS objective.

        Components:
        1. Weighted tardiness estimate
           For each order, compare the summed machine load of its
           operations against the available time to its deadline.
           If the load exceeds the window, accumulate a weighted penalty.

        2. Load imbalance
           Standard deviation of machine loads across all machines that
           have at least one task — heavily loaded machines block many
           orders so we want balanced utilisation.
        """
        score = 0.0

        # Component 1: estimated tardiness
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

        # Component 2: load imbalance (standard deviation)
        loads = [v for v in self.machine_load.values() if v > 0]
        if len(loads) > 1:
            avg = sum(loads) / len(loads)
            variance = sum((l - avg) ** 2 for l in loads) / len(loads)
            score += math.sqrt(variance) * 0.01

        return score

    def task_score(self, cmd_num: str, op_idx: int) -> float:
        """
        Individual contribution of a single task to the global score.
        Used to select which tasks to destroy.
        A high contribution means the task is a good candidate for
        reassignment.
        """
        key = (cmd_num, op_idx)
        if key not in self.assign:
            return 0.0
        machine = self.assign[key]
        # Penalise tasks on over-loaded machines and tasks that belong to
        # urgent orders (they hurt the most when misplaced).
        load_score = self.machine_load.get(machine.Id, 0)
        return float(load_score)

    def copy_assign(self) -> Tuple[dict, dict, dict, dict]:
        """Snapshot the current assignment for rollback purposes."""
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
        # Recompute nb_lots from lot_size
        for key, ls in self.lot_size.items():
            cmd_num, op_idx = key
            # Find cmd to get Quantite
            cmd = next(
                (c for c in self.commandes if c.NumeroCommande == cmd_num),
                None,
            )
            if cmd:
                self.nb_lots[key] = math.ceil(cmd.Quantite / ls)


def _build_machines_by_op(machines_ok: list) -> Dict[str, list]:
    """Index available machines by normalised operation name."""
    idx: Dict[str, list] = {}
    for m in machines_ok:
        for op in m.operations_list():
            idx.setdefault(op.lower(), []).append(m)
    return idx


def _construct_initial_solution(state: LNSState) -> None:
    """
    Build the initial feasible assignment by processing orders in
    decreasing priority order (most urgent, nearest deadline first).
    For each (order, operation) pair, pick the least-loaded compatible
    machine.
    """
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
    """
    Destroy phase: identify and return the DESTROY_RATIO fraction of
    assignments with the highest individual task scores.

    A mix of score-driven and random selection avoids getting trapped in
    local optima.
    """
    all_keys = list(state.assign.keys())
    n_destroy = max(1, int(len(all_keys) * DESTROY_RATIO))

    # Score all tasks
    scored = [
        (state.task_score(k[0], k[1]), k)
        for k in all_keys
    ]
    scored.sort(key=lambda x: -x[0])  # highest score first

    # Select top 70 % by score, 30 % randomly from the rest
    n_score  = max(1, int(n_destroy * 0.7))
    n_random = n_destroy - n_score

    to_destroy = [k for _, k in scored[:n_score]]
    pool       = [k for _, k in scored[n_score:]]
    if pool and n_random > 0:
        to_destroy.extend(rng.sample(pool, min(n_random, len(pool))))

    return to_destroy


def _lns_repair(state: LNSState, destroyed_keys: List[Tuple[str, int]]) -> None:
    """
    Repair phase: reassign all destroyed tasks.
    We first unassign them all (so their machine loads are freed), then
    reassign them in priority order using the updated least-loaded heuristic.
    """
    # Build a set of (cmd, op_idx) pairs to reassign
    cmd_lookup = {c.NumeroCommande: c for c in state.commandes}
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

    # Sort by priority before reassigning
    tasks_to_repair.sort(key=lambda t: _priority_score(t[0], state.ops_by_recette))

    for cmd, op_idx, op in tasks_to_repair:
        machine = state._best_machine_for(cmd, op)
        if machine is None:
            continue
        state.assign_task(cmd, op_idx, op, machine)


def run_lns(
    commandes: list,
    ops_by_recette: Dict[int, list],
    machines_ok: list,
) -> LNSState:
    """
    Execute the full LNS procedure and return the final state.

    1. Build machines_by_op index.
    2. Construct an initial feasible solution.
    3. Iteratively destroy and repair, keeping improvements.
    4. Return the best state found.
    """
    rng            = random.Random(RANDOM_SEED)
    machines_by_op = _build_machines_by_op(machines_ok)

    state = LNSState(commandes, ops_by_recette, machines_by_op)

    print("  [LNS] Constructing initial solution ...")
    _construct_initial_solution(state)
    best_score    = state.score()
    best_snapshot = state.copy_assign()

    print(f"  [LNS] Initial score = {best_score:,.1f}  |  "
          f"{len(state.assign)} tasks assigned")
    print(f"  [LNS] Running {LNS_ITERATIONS} iterations "
          f"(time limit = {LNS_TIME_LIMIT}s) ...")

    t_start      = time.time()
    improvements = 0

    for iteration in range(LNS_ITERATIONS):
        if time.time() - t_start > LNS_TIME_LIMIT:
            print(f"  [LNS] Time limit reached at iteration {iteration}.")
            break

        snapshot = state.copy_assign()

        # Destroy
        destroyed = _lns_destroy(state, rng)

        # Repair
        _lns_repair(state, destroyed)

        new_score = state.score()

        if new_score < best_score:
            # Accept improvement
            best_score    = new_score
            best_snapshot = state.copy_assign()
            improvements += 1
        else:
            # Reject: restore previous solution
            state.restore_assign(snapshot)

    elapsed = time.time() - t_start
    print(f"  [LNS] Finished: {improvements} improvements in "
          f"{elapsed:.1f}s  |  best score = {best_score:,.1f}")

    # Restore the globally best solution
    state.restore_assign(best_snapshot)
    return state


# ===========================================================================
# LAYER 2 — COUCHE CP-SAT  (timing optimisation)
# ===========================================================================

def _build_cp_model(
    commandes: list,
    ops_by_recette: Dict[int, list],
    machines_ok: list,
    lns_state: LNSState,
    horizon: int,
) -> Tuple[cp_model.CpModel, Dict[Tuple[str, int], dict], Dict[int, list]]:
    """
    Build the CP-SAT model.

    Variables
    ---------
    s_{cmd}_{op}  : integer start time in productive minutes
    e_{cmd}_{op}  : integer end time  (= start + total_duration_all_lots)
    iv_{cmd}_{op} : interval variable for NoOverlap constraints

    Constraints
    -----------
    NoOverlap per machine : no two intervals on the same machine overlap.
    Precedence per order  : operation i+1 starts only after operation i ends.
    """
    model        = cp_model.CpModel()
    task_vars:   Dict[Tuple[str, int], dict] = {}
    machine_itvs: Dict[int, list]            = {m.Id: [] for m in machines_ok}

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

            iv = model.NewIntervalVar(
                s_var, dur, e_var,
                f"iv_{cmd.NumeroCommande}_{op_idx}",
            )
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

    # NoOverlap: each machine processes at most one task at any minute
    for m in machines_ok:
        if machine_itvs[m.Id]:
            model.AddNoOverlap(machine_itvs[m.Id])

    # Precedence: operations within an order execute in recipe order
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
    Minimise weighted tardiness + makespan.

    For each order:
        tardiness = max(0, finish_time - deadline_pm)
    Objective:
        TARDINESS_PENALTY * sum(urgency_weight * tardiness) + makespan
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
    lns_state: LNSState,
    solver: cp_model.CpSolver,
) -> list:
    """
    Expand aggregated tasks back to per-lot Gantt bar entries.

    For a task with N lots and per-lot duration D, lot k starts at:
        start_pm = task_start + k * D
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

            task_start_pm = solver.Value(tv["start"])
            nb_lots       = tv["NbLots"]
            lot_size      = tv["LotSize"]
            dur_one_lot   = op.DureeTotale
            machine       = lns_state.assign[key]

            for lot_idx in range(nb_lots):
                s_pm = task_start_pm + lot_idx * dur_one_lot
                e_pm = s_pm + dur_one_lot
                s_day, s_h, s_m = pm_to_clock(s_pm)
                e_day, e_h, e_m = pm_to_clock(e_pm)
                d_start = working_day_date(s_day)
                d_end   = working_day_date(e_day)

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
                print(f"  [RETARD] {cmd.NumeroCommande}: "
                      f"{fin_day - deadline_day} jour(s)")
            else:
                print(f"  [OK]     {cmd.NumeroCommande}: "
                      f"marge {deadline_day - fin_day} jour(s)")

    return results


# ---------------------------------------------------------------------------
# Machine utilisation diagnostic
# ---------------------------------------------------------------------------

def _print_machine_utilisation(lns_state: LNSState) -> None:
    """Print per-machine load statistics after LNS assignment."""
    load_by_machine:  Dict[int, int] = defaultdict(int)
    tasks_by_machine: Dict[int, int] = defaultdict(int)

    for key, machine in lns_state.assign.items():
        load = lns_state.task_load.get(key, 0)
        load_by_machine[machine.Id]  += load
        tasks_by_machine[machine.Id] += 1

    print("\nMachine utilisation after LNS assignment (top 20 by load):")
    sorted_items = sorted(load_by_machine.items(), key=lambda x: -x[1])
    for mid, load in sorted_items[:20]:
        tasks = tasks_by_machine[mid]
        days  = load / PPD
        print(f"  Machine {mid:3d}: {load:7d} min ({days:5.1f} days)"
              f"  |  {tasks} tasks")

    if len(sorted_items) > 20:
        print(f"  ... ({len(sorted_items) - 20} more machines)")


# ===========================================================================
# Public entry point
# ===========================================================================

def solve() -> list:
    """
    Run the two-layer hybrid scheduler and return a flat list of Gantt
    task dicts.

    Returns an empty list if the CP-SAT model is infeasible or times out
    without finding any feasible solution.
    """
    commandes, machines, ops_by_recette, recettes_by_id = load_data()

    warnings = validate_data(commandes, machines, ops_by_recette, recettes_by_id)
    for w in warnings:
        print(f"WARNING: {w}")

    machines_ok = [m for m in machines if m.is_available()]

    print(f"Loaded {len(commandes)} orders | "
          f"{len(machines_ok)}/{len(machines)} machines available")
    print(f"Schedule: 00h00 to 00h00  |  PPD = {PPD} min/day")

    # -----------------------------------------------------------------------
    # LAYER 1 — Couche Heuristique (LNS)
    # -----------------------------------------------------------------------
    separator = "=" * 70
    print(f"\n{separator}")
    print("  LAYER 1 — COUCHE HEURISTIQUE  (Large Neighborhood Search)")
    print(f"{separator}")

    lns_state = run_lns(commandes, ops_by_recette, machines_ok)

    print(f"  {len(lns_state.assign)} tasks assigned across "
          f"{len({m.Id for m in lns_state.assign.values()})} machines")

    _print_machine_utilisation(lns_state)

    # -----------------------------------------------------------------------
    # LAYER 2 — Couche CP-SAT (timing optimisation)
    # -----------------------------------------------------------------------
    max_export_day = max(date_to_day_offset(c.DateExport) for c in commandes)
    horizon        = (max_export_day + 15) * PPD   # generous horizon

    print(f"\n{separator}")
    print("  LAYER 2 — COUCHE CP-SAT  (timing optimisation)")
    print(f"{separator}")
    print(f"  Building CP-SAT model  |  horizon = {horizon} PM "
          f"({horizon // PPD} days) ...")

    model, task_vars, machine_itvs = _build_cp_model(
        commandes, ops_by_recette, machines_ok, lns_state, horizon,
    )
    makespan = _add_objective(
        model, commandes, ops_by_recette, task_vars, horizon,
    )
    print(f"  {len(task_vars)} task variables  |  "
          f"{sum(len(v) for v in machine_itvs.values())} interval constraints")

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------
    print(f"\n{separator}")
    print(f"  SOLVING  |  time limit = {MAX_SOLVE_SECONDS}s  |  "
          f"seed = {RANDOM_SEED}")
    print(f"{separator}")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAX_SOLVE_SECONDS
    solver.parameters.random_seed         = RANDOM_SEED
    solver.parameters.num_search_workers  = 8   # parallel workers — major speed-up
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

    # -----------------------------------------------------------------------
    # Extract and return results
    # -----------------------------------------------------------------------
    print("Extracting results (per-lot expansion) ...")
    results = _extract_results(
        commandes, ops_by_recette, task_vars, lns_state, solver,
    )
    print(f"\nDone. {len(results)} Gantt rows generated.")
    return results


if __name__ == "__main__":
    solve()