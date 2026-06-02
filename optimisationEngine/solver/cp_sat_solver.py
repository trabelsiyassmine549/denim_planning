import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False


from optimisationEngine.models.schemas import GanttRow
from optimisationEngine.utils.time_utils import (
    PPD,
    date_to_day_offset,
    date_to_pm,
    pm_to_clock,
    working_day_date,
)

LNS_ITERATIONS    = 300
LNS_TIME_LIMIT    = 45.0
DESTROY_RATIO     = 0.25
RANDOM_SEED       = 42
MAX_SOLVE_SECONDS = 120
TARDINESS_PENALTY = 100_000


def _urgency_weight(urgence: int) -> int:
    return max(1, 10 // max(urgence, 1))


def _priority_score(cmd, ops_by_recette: dict) -> float:
    ops = ops_by_recette.get(cmd.RecetteId, [])
    total_duration = sum(op.DureeTotale for op in ops)
    deadline_pm = date_to_pm(cmd.DateExport)
    return (
        cmd.Urgence * 1000
        - deadline_pm
        + total_duration
    )


def _effective_lot_size(cmd, op, machine) -> int:
    return max(1, min(op.QuantiteLot, machine.CapaciteMax, cmd.Quantite))

# couche LNS
class LNSState:

    def __init__(self, commandes, ops_by_recette, machines_by_op):
        self.commandes       = commandes
        self.ops_by_recette  = ops_by_recette
        self.machines_by_op  = machines_by_op  

        self.assign:       Dict[Tuple[str, int], List[object]] = {}
        self.machine_load: Dict[int, int]                       = defaultdict(int)
        self.task_load:    Dict[Tuple[str, int], int]           = {}
        self.lot_size:     Dict[Tuple[str, int], int]           = {}
        self.nb_lots:      Dict[Tuple[str, int], int]           = {}
        self.machine_lots: Dict[Tuple[str, int], List[Tuple[object, int]]] = {}


    def _compute_task_load(self, cmd, op, machine) -> int:
        ls = _effective_lot_size(cmd, op, machine)
        nb = math.ceil(cmd.Quantite / ls)
        return op.DureeTotale * nb

    def _best_machine_for(self, cmd, op):
        op_key     = op.NomOperation.lower()
        candidates = self.machines_by_op.get(op_key, [])
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda m: self.machine_load[m.Id] + self._compute_task_load(cmd, op, m)
        )

    def _best_n_machines_for(self, op, n: int) -> List[object]:
       
        op_key = op.NomOperation.lower()
        candidates = self.machines_by_op.get(op_key, [])
        if not candidates:
            return []
        sorted_candidates = sorted(candidates, key=lambda m: self.machine_load[m.Id])
        return sorted_candidates[:n]

    def assign_task(self, cmd, op_idx: int, op, machines_list: List[object]) -> None:
    
        if not machines_list:
            return

        key = (cmd.NumeroCommande, op_idx)

        if key in self.assign:
            for old_m, old_nb in self.machine_lots.get(key, []):
                self.machine_load[old_m.Id] -= op.DureeTotale * old_nb

        primary_machine = machines_list[0]
        ls       = _effective_lot_size(cmd, op, primary_machine)
        total_nb = max(1, math.ceil(cmd.Quantite / ls))

        n_machines = len(machines_list)
        base_lots, remainder = divmod(total_nb, n_machines)
        distribution: List[Tuple[object, int]] = []
        for i, m in enumerate(machines_list):
            nb = base_lots + (1 if i < remainder else 0)
            if nb > 0:
                distribution.append((m, nb))
                self.machine_load[m.Id] += op.DureeTotale * nb

        self.assign[key]       = machines_list
        self.machine_lots[key] = distribution
        self.lot_size[key]     = ls
        self.nb_lots[key]      = total_nb
        self.task_load[key]    = op.DureeTotale * total_nb

    def _unassign_with_op(self, cmd_num: str, op_idx: int, op) -> Optional[List[object]]:

        key = (cmd_num, op_idx)
        if key not in self.assign:
            return None
        machines = self.assign.pop(key)
        for m, nb in self.machine_lots.pop(key, []):
            self.machine_load[m.Id] -= op.DureeTotale * nb
        self.task_load.pop(key, None)
        self.lot_size.pop(key, None)
        self.nb_lots.pop(key, None)
        return machines

    def unassign_task(self, cmd_num: str, op_idx: int) -> Optional[List[object]]:

        key = (cmd_num, op_idx)
        if key not in self.assign:
            return None
        machines  = self.assign.pop(key)
        task_load = self.task_load.pop(key, 0)
        nb_lots   = self.nb_lots.pop(key, 1)
        for m, nb in self.machine_lots.pop(key, []):
            per_lot = (task_load // nb_lots) if nb_lots > 0 else 0
            self.machine_load[m.Id] -= per_lot * nb
        self.lot_size.pop(key, None)
        return machines


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

    def copy_assign(self):
        return (
            {k: list(v) for k, v in self.assign.items()},
            dict(self.machine_load),
            dict(self.task_load),
            dict(self.lot_size),
            dict(self.nb_lots),
            {k: list(v) for k, v in self.machine_lots.items()},
        )

    def restore_assign(self, snapshot) -> None:
        assign, machine_load, task_load, lot_size, nb_lots, machine_lots = snapshot
        self.assign       = {k: list(v) for k, v in assign.items()}
        self.machine_load = defaultdict(int, machine_load)
        self.task_load    = dict(task_load)
        self.lot_size     = dict(lot_size)
        self.nb_lots      = dict(nb_lots)
        self.machine_lots = {k: list(v) for k, v in machine_lots.items()}


def _build_machines_by_op(machines_ok: list) -> Dict[str, list]:
    idx: Dict[str, list] = {}
    for m in machines_ok:
        for op in m.operations_list():
            idx.setdefault(op.lower(), []).append(m)
    return idx


def _construct_initial_solution(state: LNSState, max_machines_per_op: int = 1) -> None:
    sorted_cmds = sorted(
        state.commandes,
        key=lambda c: _priority_score(c, state.ops_by_recette),
    )
    for cmd in sorted_cmds:
        ops = state.ops_by_recette.get(cmd.RecetteId, [])
        for op_idx, op in enumerate(ops):
            machines = state._best_n_machines_for(op, max_machines_per_op)
            if not machines:
                continue
            state.assign_task(cmd, op_idx, op, machines)


def _lns_destroy(state: LNSState) -> List[Tuple[str, int]]:
    all_keys  = list(state.assign.keys())
    n_destroy = max(1, int(len(all_keys) * DESTROY_RATIO))
    scored = sorted(
        all_keys,
        key=lambda k: -state.machine_load.get(
            state.assign[k][0].Id if state.assign[k] else 0, 0
        ),
    )
    return scored[:n_destroy]


def _lns_repair(state: LNSState, destroyed_keys: List[Tuple[str, int]],
                max_machines_per_op: int = 1) -> None:
    cmd_lookup = {c.NumeroCommande: c for c in state.commandes}

    tasks_to_repair = []
    for key in destroyed_keys:
        cmd_num, op_idx = key
        cmd = cmd_lookup.get(cmd_num)
        if cmd is None:
            continue
        ops = state.ops_by_recette.get(cmd.RecetteId, [])
        if op_idx >= len(ops):
            continue
        op = ops[op_idx]
        state._unassign_with_op(cmd_num, op_idx, op)
        tasks_to_repair.append((cmd, op_idx, op))

    for cmd, op_idx, op in tasks_to_repair:
        machines = state._best_n_machines_for( op, max_machines_per_op)
        if not machines:
            continue
        state.assign_task(cmd, op_idx, op, machines)


def run_lns(commandes, ops_by_recette, machines_ok,
            max_machines_per_op: int = 1) -> LNSState:
    machines_by_op = _build_machines_by_op(machines_ok)
    state          = LNSState(commandes, ops_by_recette, machines_by_op)

    print(f"  [LNS] Constructing initial solution (max {max_machines_per_op} machine(s)/op) ...")
    _construct_initial_solution(state, max_machines_per_op)
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
        _lns_repair(state, destroyed, max_machines_per_op)
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

# couche cp-sat

def _build_cp_model(commandes, ops_by_recette, machines_ok, lns_state, horizon):
    model        = cp_model.CpModel()
    task_vars    = {}
    machine_itvs = {m.Id: [] for m in machines_ok}

    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for op_idx, op in enumerate(ops):
            key = (cmd.NumeroCommande, op_idx)
            if key not in lns_state.assign:
                continue

            distribution = lns_state.machine_lots.get(key, [])
            if not distribution:
                continue

            lot_size = lns_state.lot_size.get(key, op.QuantiteLot)
            nb_lots  = lns_state.nb_lots.get(key, 1)

            safe_cmd = "".join(c if c.isalnum() or c == "_" else "_"
                               for c in cmd.NumeroCommande)

            slice_vars = []
            for slice_idx, (machine, slice_nb) in enumerate(distribution):
                dur = op.DureeTotale * slice_nb

                if dur <= 0 or dur >= horizon:
                    continue

                s_ub = max(0, horizon - dur)
                s_var = model.NewIntVar(0, s_ub,
                    f"s_{safe_cmd}_{op_idx}_{slice_idx}")
                e_var = model.NewIntVar(dur, horizon,
                    f"e_{safe_cmd}_{op_idx}_{slice_idx}")
                model.Add(e_var == s_var + dur)

                iv = model.NewIntervalVar(s_var, dur, e_var,
                    f"iv_{safe_cmd}_{op_idx}_{slice_idx}")
                machine_itvs[machine.Id].append(iv)

                slice_vars.append({
                    "start":   s_var,
                    "end":     e_var,
                    "machine": machine,
                    "nb_lots": slice_nb,
                })

            if not slice_vars:
                continue

            task_vars[key] = {
                "slices":                   slice_vars,
                "NomOperation":             op.NomOperation,
                "DureeMinutes":             op.DureeMinutes,
                "TempsChargementMinutes":   op.TempsChargementMinutes,
                "TempsDecharementMinutes":  op.TempsDecharementMinutes,
                "DureeTotale":              op.DureeTotale,
                "LotSize":                  lot_size,
                "NbLots":                   nb_lots,
            }

    for m in machines_ok:
        if machine_itvs[m.Id]:
            model.AddNoOverlap(machine_itvs[m.Id])

    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for i in range(len(ops) - 1):
            k0 = (cmd.NumeroCommande, i)
            k1 = (cmd.NumeroCommande, i + 1)
            if k0 not in task_vars or k1 not in task_vars:
                continue
            ends0   = [sv["end"]   for sv in task_vars[k0]["slices"]]
            starts1 = [sv["start"] for sv in task_vars[k1]["slices"]]
            if not ends0 or not starts1:
                continue
            safe_cmd_p = "".join(c if c.isalnum() or c == "_" else "_"
                                 for c in cmd.NumeroCommande)
            max_end0 = model.NewIntVar(0, horizon, f"maxend_{safe_cmd_p}_{i}")
            model.AddMaxEquality(max_end0, ends0)
            for s1 in starts1:
                model.Add(s1 >= max_end0)

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

        slices   = task_vars[last_key]["slices"]
        end_vars = [sv["end"] for sv in slices]
        if not end_vars:
            continue

        safe_cmd_o = "".join(c if c.isalnum() or c == "_" else "_"
                              for c in cmd.NumeroCommande)
        if len(end_vars) == 1:
            end_var = end_vars[0]
        else:
            end_var = model.NewIntVar(0, horizon, f"cmdend_{safe_cmd_o}")
            model.AddMaxEquality(end_var, end_vars)

        deadline = date_to_pm(cmd.DateExport)
        weight   = _urgency_weight(cmd.Urgence)

        late = model.NewIntVar(-horizon, horizon, f"late_{safe_cmd_o}")
        tard = model.NewIntVar(0,        horizon, f"tard_{safe_cmd_o}")
        model.Add(late == end_var - deadline)
        model.AddMaxEquality(tard, [late, model.NewConstant(0)])
        tard_terms.append((weight, tard))
        all_ends.append(end_var)

    makespan = model.NewIntVar(0, horizon, "makespan")
    if all_ends:
        model.AddMaxEquality(makespan, all_ends)
    else:
        model.Add(makespan == 0)

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

            lot_size  = tv["LotSize"]
            nb_lots   = tv["NbLots"]
            slices    = tv["slices"]

            global_lot_idx = 0
            for sv in slices:
                slice_start_pm = solver.Value(sv["start"])
                slice_nb       = sv["nb_lots"]
                machine        = sv["machine"]

                for local_idx in range(slice_nb):
                    s_pm  = slice_start_pm + local_idx * op.DureeTotale
                    e_pm  = s_pm + op.DureeTotale
                    s_day = pm_to_clock(s_pm)[0]
                    e_day = pm_to_clock(e_pm)[0]

                    if global_lot_idx == nb_lots - 1:
                        pieces = cmd.Quantite - global_lot_idx * lot_size
                    else:
                        pieces = lot_size

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
                        lotSize                 = max(1, pieces),
                        quantiteLot             = lot_size,
                        lotIdx                  = global_lot_idx,
                        nbLots                  = nb_lots,
                        dateStart               = working_day_date(s_day).isoformat(),
                        dateEnd                 = working_day_date(e_day).isoformat(),
                        dateExport              = cmd.DateExport,
                    ))
                    global_lot_idx += 1

    return rows


def _print_machine_utilisation(lns_state: LNSState) -> None:
    load_by_machine  = defaultdict(int)
    tasks_by_machine = defaultdict(int)

    for key, dist in lns_state.machine_lots.items():
        for m, nb in dist:
            load_by_machine[m.Id]  += lns_state.task_load.get(key, 0)
            tasks_by_machine[m.Id] += 1

    print("\nMachine utilisation after LNS assignment (top 20 by load):")
    sorted_items = sorted(load_by_machine.items(), key=lambda x: -x[1])
    for mid, load in sorted_items[:20]:
        tasks = tasks_by_machine[mid]
        days  = load / PPD
        print(f"  Machine {mid:3d}: {load:7d} min ({days:5.1f} days)  |  {tasks} tasks")
    if len(sorted_items) > 20:
        print(f"  ... ({len(sorted_items) - 20} more machines)")



def collect_split_warnings(commandes, ops_by_recette, lns_state: "LNSState",
                            requested_machines: int) -> List[str]:
    

    if not hasattr(lns_state, "machines_by_op") or not lns_state.machines_by_op:
        return []


    limited_ops: Dict[str, Dict[str, int]] = {}

    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for op_idx, op in enumerate(ops):
            key = (cmd.NumeroCommande, op_idx)
            distribution = lns_state.machine_lots.get(key, [])
            if not distribution:
                continue
            actual = len(distribution)
            if actual < requested_machines:
                op_key    = op.NomOperation.lower()
                available = len(lns_state.machines_by_op.get(op_key, []))
                op_name   = op.NomOperation
                if op_name not in limited_ops:
                    limited_ops[op_name] = {"available": available, "actual": actual}
                else:
                    if actual < limited_ops[op_name]["actual"]:
                        limited_ops[op_name]["actual"]    = actual
                        limited_ops[op_name]["available"] = available

    if not limited_ops:
        return []

    warnings: List[str] = []


    for op_name, info in sorted(limited_ops.items()):
        actual    = info["actual"]
        available = info["available"]
        if available < requested_machines:
            kind = "NOT_ENOUGH_MACHINES"
        else:
            kind = "NOT_ENOUGH_LOTS"
        warnings.append(
            f"SPLIT_WARNING|{kind}|{op_name}|{actual}|{available}|{requested_machines}"
        )
    return warnings


# Public API : used by api.py

class SolverResult:
    def __init__(self, solver, task_vars, makespan_pm: int):
        self._solver     = solver
        self._task_vars  = task_vars
        self.makespan_pm = makespan_pm

    def extract_rows(self, commandes, ops_by_recette, lns_state) -> List[GanttRow]:
        return _extract_results(commandes, ops_by_recette, self._task_vars, lns_state, self._solver)

# horizon c'est la durée max de la planif, en PM.  
def run_cpsat(commandes, ops_by_recette, machines_ok, lns_state, horizon) -> SolverResult:
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
        raise RuntimeError(
            f"CP-SAT n'a pas trouvé de solution faisable "
            f"(status={solver.StatusName(status)}). "
            f"Vérifiez que les machines et recettes sont cohérentes."
        )

    return SolverResult(solver, task_vars, solver.Value(makespan_var))


def lns_fallback(commandes, ops_by_recette, lns_state) -> List[GanttRow]:

    machine_end: Dict[int, int] = defaultdict(int)
    rows: List[GanttRow] = []

    for cmd in sorted(commandes, key=lambda c: c.Urgence):
        ops      = ops_by_recette.get(cmd.RecetteId, [])
        prev_end = 0

        for op_idx, op in enumerate(ops):
            key          = (cmd.NumeroCommande, op_idx)
            distribution = lns_state.machine_lots.get(key, [])
           
            if not distribution:
                continue

            nb_lots  = lns_state.nb_lots.get(key, 1)
            lot_size = lns_state.lot_size.get(key, op.QuantiteLot)

            op_end         = prev_end
            global_lot_idx = 0

            for machine, slice_nb in distribution:
                s_pm = max(machine_end[machine.Id], prev_end)
                e_pm = s_pm + op.DureeTotale * slice_nb
                machine_end[machine.Id] = e_pm
                op_end = max(op_end, e_pm)

                for local_idx in range(slice_nb):
                    sp    = s_pm + local_idx * op.DureeTotale
                    ep    = sp + op.DureeTotale
                    s_day = pm_to_clock(sp)[0]
                    e_day = pm_to_clock(ep)[0]
                    if global_lot_idx == nb_lots - 1:
                        pieces = cmd.Quantite - global_lot_idx * lot_size
                    else:
                        pieces = lot_size

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
                        lotSize                 = max(1, pieces),
                        quantiteLot             = lot_size,
                        lotIdx                  = global_lot_idx,
                        nbLots                  = nb_lots,
                        dateStart               = working_day_date(s_day).isoformat(),
                        dateEnd                 = working_day_date(e_day).isoformat(),
                        dateExport              = cmd.DateExport,
                    ))
                    global_lot_idx += 1

            prev_end = op_end

    return rows

def solve(max_machines_per_op: int = 1) -> List[GanttRow]:
    from utils.data_loader import load_data, validate_data   

    commandes, machines, ops_by_recette, recettes_by_id = load_data()

    for w in validate_data(commandes, machines, ops_by_recette, recettes_by_id):
        print(f"WARNING: {w}")

    machines_ok = [m for m in machines if m.is_available()]
    print(
        f"Loaded {len(commandes)} orders | "
        f"{len(machines_ok)}/{len(machines)} machines available | "
        f"max {max_machines_per_op} machine(s)/op"
    )

    sep = "=" * 70
    print(f"\n{sep}\n  LAYER 1 — COUCHE HEURISTIQUE  (Large Neighborhood Search)\n{sep}")

    lns_state = run_lns(commandes, ops_by_recette, machines_ok,
                        max_machines_per_op=max_machines_per_op)
    _print_machine_utilisation(lns_state)

    max_export_day = max(date_to_day_offset(c.DateExport) for c in commandes)
    horizon        = max((max_export_day + 15) * PPD, 1000)

    print(f"\n{sep}\n  LAYER 2 — COUCHE CP-SAT  (timing optimisation)\n{sep}")
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
    rows = result.extract_rows(commandes, ops_by_recette, lns_state)
    print(f"\nDone. {len(rows)} Gantt rows generated.")
    return rows


if __name__ == "__main__":
    solve()