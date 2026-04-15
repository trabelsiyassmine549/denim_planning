"""
api.py  —  FastAPI bridge for the Denim Washing Production Planner
==================================================================
Reads live data from the .NET backend (CommandesDB via REST),
runs the CP-SAT optimiser and returns structured Gantt rows.

Run:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /api/planning/run        → run optimiser, return Gantt rows
    GET  /api/planning/health     → liveness check
"""

import math
import random
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Optional OR-Tools import (graceful degradation) ────────────────────────
try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False

# ===========================================================================
# Configuration
# ===========================================================================

DOTNET_BASE_URL   = "https://localhost:7228/api"   # change port if needed
LNS_ITERATIONS    = 200
LNS_TIME_LIMIT    = 30.0
DESTROY_RATIO     = 0.25
RANDOM_SEED       = 42
MAX_SOLVE_SECONDS = 60
TARDINESS_PENALTY = 100_000
PPD               = 1440                           # productive minutes per day

app = FastAPI(title="Denim Planner Optimiser", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "https://localhost:4200"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

# ===========================================================================
# Pydantic schemas (request / response)
# ===========================================================================

class RunRequest(BaseModel):
    token: str                          # Bearer JWT from Angular
    commandeIds: Optional[List[int]] = None   # None → all "En attente"


class GanttRow(BaseModel):
    numeroCommande:          str
    quantite:                int
    recetteId:               int
    urgence:                 int
    nomOperation:            str
    machineId:               int
    machineName:             str
    startPM:                 int
    endPM:                   int
    dureeMinutes:            int
    tempsChargementMinutes:  int
    tempsDecharementMinutes: int
    dureeTotale:             int
    lotSize:                 int
    quantiteLot:             int
    lotIdx:                  int
    nbLots:                  int
    dateStart:               str
    dateEnd:                 str
    dateExport:              str


class RunResponse(BaseModel):
    status:        str
    makespanDays:  int
    makespanPM:    int
    startDate:     str
    rows:          List[GanttRow]
    warnings:      List[str]


# ===========================================================================
# Domain models (internal)
# ===========================================================================

class Commande:
    def __init__(self, d: dict):
        self.Id             = d["id"]
        self.NumeroCommande = d["numeroCommande"]
        self.DateExport     = d["dateExport"][:10]
        self.Urgence        = d["urgence"]
        self.Quantite       = d["quantite"]
        self.RecetteId      = d["recetteId"]


class Machine:
    def __init__(self, d: dict):
        self.Id         = d["id"]
        self.NomMachine = d["nomMachine"]
        self.CapaciteMax = d["capaciteMax"]
        self.Statut     = d["statut"]
        self.Operations = d.get("operations") or ""

    def operations_list(self):
        return [op.strip() for op in self.Operations.split(",") if op.strip()]

    def is_available(self):
        return self.Statut.strip().lower() == "fonctionnel"

    def supports_operation(self, op: str):
        return op.lower() in [o.lower() for o in self.operations_list()]


class OperationRecette:
    def __init__(self, d: dict):
        self.Id                      = d["id"]
        self.RecetteId               = d.get("recetteId", 0)
        self.Ordre                   = d["ordre"]
        self.NomOperation            = d["nomOperation"]
        self.DureeMinutes            = d["dureeMinutes"]
        self.QuantiteLot             = d["quantiteLot"]
        self.TempsChargementMinutes  = d.get("tempsChargementMinutes", 5)
        self.TempsDecharementMinutes = d.get("tempsDecharementMinutes", 5)

    @property
    def DureeTotale(self):
        return self.TempsChargementMinutes + self.DureeMinutes + self.TempsDecharementMinutes


# ===========================================================================
# Time utilities
# ===========================================================================

START_DATE = date.today()

def _advance_to_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d

START_DATE = _advance_to_weekday(START_DATE)


def working_day_date(offset: int) -> date:
    if offset <= 0:
        return START_DATE
    d, n = START_DATE, 0
    while n < offset:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return d


def date_to_day_offset(iso: str) -> int:
    target = date.fromisoformat(iso)
    if target <= START_DATE:
        return 1
    d, count = START_DATE, 0
    while d < target:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return max(count, 1)


def date_to_pm(iso: str) -> int:
    return date_to_day_offset(iso) * PPD


def pm_to_clock(pm: int):
    day = pm // PPD
    off = pm % PPD
    return day, off // 60, off % 60


# ===========================================================================
# Data fetching from .NET
# ===========================================================================

async def _fetch(client: httpx.AsyncClient, path: str, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    r = await client.get(f"{DOTNET_BASE_URL}{path}", headers=headers, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code,
                            detail=f"Backend error on {path}: {r.text[:200]}")
    return r.json()


async def load_live_data(token: str, commande_ids: Optional[List[int]] = None):
    """Fetch commandes, machines, recettes+operations from .NET."""
    async with httpx.AsyncClient(verify=False) as client:
        raw_cmds     = await _fetch(client, "/Commandes", token)
        raw_machines = await _fetch(client, "/Machines", token)
        raw_recettes = await _fetch(client, "/Recettes", token)

    # Filter commandes
    commandes = [Commande(c) for c in raw_cmds
                 if c.get("statut", "").lower() == "en attente"]
    if commande_ids:
        commandes = [c for c in commandes if c.Id in commande_ids]

    machines = [Machine(m) for m in raw_machines]

    # Build ops_by_recette from embedded recette.operations
    ops_by_recette: Dict[int, List[OperationRecette]] = {}
    for r in raw_recettes:
        rid = r["id"]
        ops = [OperationRecette({**op, "recetteId": rid})
               for op in r.get("operations", [])]
        ops.sort(key=lambda o: o.Ordre)
        ops_by_recette[rid] = ops

    return commandes, machines, ops_by_recette


# ===========================================================================
# LNS (Layer 1)
# ===========================================================================

def _eff_lot(cmd, op, machine):
    return min(op.QuantiteLot, machine.CapaciteMax, cmd.Quantite)


def _urgency_weight(u: int) -> int:
    return max(1, 10 // max(u, 1))


def _priority_score(cmd, ops_by_recette):
    ops   = ops_by_recette.get(cmd.RecetteId, [])
    total = sum(op.DureeTotale for op in ops)
    dl    = date_to_pm(cmd.DateExport)
    return cmd.Urgence * 1000 - dl + total


def _build_machines_by_op(machines_ok):
    idx = {}
    for m in machines_ok:
        for op in m.operations_list():
            idx.setdefault(op.lower(), []).append(m)
    return idx


class LNSState:
    def __init__(self, commandes, ops_by_recette, machines_by_op):
        self.commandes      = commandes
        self.ops_by_recette = ops_by_recette
        self.machines_by_op = machines_by_op
        self.assign:       dict = {}
        self.machine_load: dict = defaultdict(int)
        self.task_load:    dict = {}
        self.lot_size:     dict = {}
        self.nb_lots:      dict = {}

    def _task_load(self, cmd, op, machine):
        ls = _eff_lot(cmd, op, machine)
        return op.DureeTotale * math.ceil(cmd.Quantite / ls)

    def _best_machine(self, cmd, op):
        cands = self.machines_by_op.get(op.NomOperation.lower(), [])
        if not cands:
            return None
        return min(cands, key=lambda m: self.machine_load[m.Id] + self._task_load(cmd, op, m))

    def assign_task(self, cmd, op_idx, op, machine):
        key = (cmd.NumeroCommande, op_idx)
        if key in self.assign:
            old = self.assign[key]
            self.machine_load[old.Id] -= self.task_load.get(key, 0)
        load = self._task_load(cmd, op, machine)
        ls   = _eff_lot(cmd, op, machine)
        self.assign[key]      = machine
        self.task_load[key]   = load
        self.lot_size[key]    = ls
        self.nb_lots[key]     = math.ceil(cmd.Quantite / ls)
        self.machine_load[machine.Id] += load

    def unassign_task(self, cmd_num, op_idx):
        key = (cmd_num, op_idx)
        if key not in self.assign:
            return None
        m = self.assign.pop(key)
        self.machine_load[m.Id] -= self.task_load.pop(key, 0)
        self.lot_size.pop(key, None)
        self.nb_lots.pop(key, None)
        return m

    def score(self):
        s = 0.0
        for cmd in self.commandes:
            ops = self.ops_by_recette.get(cmd.RecetteId, [])
            total = sum(self.task_load.get((cmd.NumeroCommande, i), 0) for i in range(len(ops)))
            dl    = date_to_pm(cmd.DateExport)
            s    += _urgency_weight(cmd.Urgence) * max(0, total - dl)
        loads = [v for v in self.machine_load.values() if v > 0]
        if len(loads) > 1:
            avg = sum(loads) / len(loads)
            s  += math.sqrt(sum((l - avg)**2 for l in loads) / len(loads)) * 0.01
        return s

    def snapshot(self):
        return (dict(self.assign), defaultdict(int, self.machine_load),
                dict(self.task_load), dict(self.lot_size))

    def restore(self, snap):
        a, ml, tl, ls = snap
        self.assign       = dict(a)
        self.machine_load = defaultdict(int, ml)
        self.task_load    = dict(tl)
        self.lot_size     = dict(ls)
        for key, lsz in ls.items():
            cmd_num, _ = key
            cmd = next((c for c in self.commandes if c.NumeroCommande == cmd_num), None)
            if cmd:
                self.nb_lots[key] = math.ceil(cmd.Quantite / lsz)


def run_lns(commandes, ops_by_recette, machines_ok):
    rng  = random.Random(RANDOM_SEED)
    mbo  = _build_machines_by_op(machines_ok)
    state = LNSState(commandes, ops_by_recette, mbo)

    # Initial greedy construction
    for cmd in sorted(commandes, key=lambda c: _priority_score(c, ops_by_recette)):
        for i, op in enumerate(ops_by_recette.get(cmd.RecetteId, [])):
            m = state._best_machine(cmd, op)
            if m:
                state.assign_task(cmd, i, op, m)

    best_score = state.score()
    best_snap  = state.snapshot()
    t0 = time.time()

    for _ in range(LNS_ITERATIONS):
        if time.time() - t0 > LNS_TIME_LIMIT:
            break
        keys    = list(state.assign.keys())
        n_dest  = max(1, int(len(keys) * DESTROY_RATIO))
        scored  = sorted(keys, key=lambda k: -state.machine_load.get(
            state.assign.get(k, type('', (), {'Id': -1})()).Id, 0))
        destroy = scored[:n_dest]

        snap = state.snapshot()
        cmd_lu = {c.NumeroCommande: c for c in commandes}
        for k in destroy:
            state.unassign_task(k[0], k[1])
        for k in destroy:
            cmd = cmd_lu.get(k[0])
            if not cmd:
                continue
            ops = ops_by_recette.get(cmd.RecetteId, [])
            if k[1] >= len(ops):
                continue
            m = state._best_machine(cmd, ops[k[1]])
            if m:
                state.assign_task(cmd, k[1], ops[k[1]], m)

        ns = state.score()
        if ns < best_score:
            best_score = ns
            best_snap  = state.snapshot()
        else:
            state.restore(snap)

    state.restore(best_snap)
    return state


# ===========================================================================
# CP-SAT (Layer 2)
# ===========================================================================

def run_cpsat(commandes, ops_by_recette, machines_ok, lns_state, horizon):
    if not HAS_ORTOOLS:
        raise RuntimeError("OR-Tools not installed")

    model        = cp_model.CpModel()
    task_vars    = {}
    machine_ivs  = {m.Id: [] for m in machines_ok}

    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for i, op in enumerate(ops):
            key = (cmd.NumeroCommande, i)
            if key not in lns_state.assign:
                continue
            machine  = lns_state.assign[key]
            nb_lots  = lns_state.nb_lots[key]
            dur      = op.DureeTotale * nb_lots

            s = model.NewIntVar(0,       horizon - dur, f"s_{cmd.NumeroCommande}_{i}")
            e = model.NewIntVar(dur,     horizon,       f"e_{cmd.NumeroCommande}_{i}")
            model.Add(e == s + dur)
            iv = model.NewIntervalVar(s, dur, e, f"iv_{cmd.NumeroCommande}_{i}")
            machine_ivs[machine.Id].append(iv)
            task_vars[key] = {
                "start": s, "end": e,
                "op":    op, "machine": machine,
                "nb_lots": nb_lots,
                "lot_size": lns_state.lot_size[key],
            }

    for m in machines_ok:
        if machine_ivs[m.Id]:
            model.AddNoOverlap(machine_ivs[m.Id])

    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        for i in range(len(ops) - 1):
            k0, k1 = (cmd.NumeroCommande, i), (cmd.NumeroCommande, i + 1)
            if k0 in task_vars and k1 in task_vars:
                model.Add(task_vars[k1]["start"] >= task_vars[k0]["end"])

    tard_terms, all_ends = [], []
    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        if not ops:
            continue
        lk = (cmd.NumeroCommande, len(ops) - 1)
        if lk not in task_vars:
            continue
        end_v = task_vars[lk]["end"]
        dl    = date_to_pm(cmd.DateExport)
        late  = model.NewIntVar(-horizon, horizon, f"late_{cmd.NumeroCommande}")
        tard  = model.NewIntVar(0,        horizon, f"tard_{cmd.NumeroCommande}")
        model.Add(late == end_v - dl)
        model.AddMaxEquality(tard, [late, model.NewConstant(0)])
        tard_terms.append((_urgency_weight(cmd.Urgence), tard))
        all_ends.append(end_v)

    makespan_var = model.NewIntVar(0, horizon, "makespan")
    if all_ends:
        model.AddMaxEquality(makespan_var, all_ends)
    model.Minimize(TARDINESS_PENALTY * sum(w * t for w, t in tard_terms) + makespan_var)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAX_SOLVE_SECONDS
    solver.parameters.random_seed         = RANDOM_SEED
    solver.parameters.num_search_workers  = 4
    solver.parameters.log_search_progress = False
    status = solver.Solve(model)

    OK = (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if status not in OK:
        raise RuntimeError("CP-SAT found no feasible solution")

    return solver, task_vars, solver.Value(makespan_var)


# ===========================================================================
# Result extraction
# ===========================================================================

def extract_results(commandes, ops_by_recette, task_vars, lns_state, solver):
    rows: List[GanttRow] = []

    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId, [])
        if not ops:
            continue
        for i, op in enumerate(ops):
            key = (cmd.NumeroCommande, i)
            tv  = task_vars.get(key)
            if not tv:
                continue
            s_pm     = solver.Value(tv["start"])
            nb_lots  = tv["nb_lots"]
            lot_size = tv["lot_size"]
            machine  = tv["machine"]

            for li in range(nb_lots):
                sp   = s_pm + li * op.DureeTotale
                ep   = sp   + op.DureeTotale
                sd, _, _ = pm_to_clock(sp)
                ed, _, _ = pm_to_clock(ep)
                pieces = (cmd.Quantite - li * lot_size
                          if li == nb_lots - 1 else lot_size)
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
                    lotIdx                  = li,
                    nbLots                  = nb_lots,
                    dateStart               = working_day_date(sd).isoformat(),
                    dateEnd                 = working_day_date(ed).isoformat(),
                    dateExport              = cmd.DateExport,
                ))
    return rows


# ===========================================================================
# Validation
# ===========================================================================

def validate(commandes, machines, ops_by_recette) -> List[str]:
    machines_ok = [m for m in machines if m.is_available()]
    available   = {op.lower() for m in machines_ok for op in m.operations_list()}
    warnings    = []
    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId)
        if not ops:
            warnings.append(f"[{cmd.NumeroCommande}] RecetteId={cmd.RecetteId} has no operations")
            continue
        for op in ops:
            if op.NomOperation.lower() not in available:
                warnings.append(
                    f"[{cmd.NumeroCommande}] Operation '{op.NomOperation}' has no available machine")
    return warnings


# ===========================================================================
# Endpoints
# ===========================================================================

@app.get("/api/planning/health")
async def health():
    return {"status": "ok", "ortools": HAS_ORTOOLS, "startDate": START_DATE.isoformat()}


@app.post("/api/planning/run", response_model=RunResponse)
async def run_planning(req: RunRequest):
    # 1. Load live data from .NET
    commandes, machines, ops_by_recette = await load_live_data(req.token, req.commandeIds)

    if not commandes:
        raise HTTPException(status_code=400, detail="No 'En attente' commandes found")

    machines_ok = [m for m in machines if m.is_available()]
    if not machines_ok:
        raise HTTPException(status_code=400, detail="No functional machines available")

    warnings = validate(commandes, machines, ops_by_recette)

    # 2. LNS
    lns_state = run_lns(commandes, ops_by_recette, machines_ok)

    if not HAS_ORTOOLS:
        # Fallback: convert LNS assignment to a simple sequential schedule
        rows = _lns_fallback(commandes, ops_by_recette, lns_state)
        ms   = max((r.endPM for r in rows), default=0)
        return RunResponse(
            status="feasible_lns_only",
            makespanDays=ms // PPD,
            makespanPM=ms,
            startDate=START_DATE.isoformat(),
            rows=rows,
            warnings=warnings + ["OR-Tools not installed; LNS fallback used"],
        )

    # 3. CP-SAT
    max_export_day = max(date_to_day_offset(c.DateExport) for c in commandes)
    horizon        = (max_export_day + 15) * PPD
    solver, task_vars, makespan_pm = run_cpsat(
        commandes, ops_by_recette, machines_ok, lns_state, horizon)

    # 4. Extract
    rows = extract_results(commandes, ops_by_recette, task_vars, lns_state, solver)

    return RunResponse(
        status="optimal" if makespan_pm < horizon else "feasible",
        makespanDays=makespan_pm // PPD,
        makespanPM=makespan_pm,
        startDate=START_DATE.isoformat(),
        rows=rows,
        warnings=warnings,
    )


def _lns_fallback(commandes, ops_by_recette, lns_state) -> List[GanttRow]:
    """Greedy sequential schedule when OR-Tools is unavailable."""
    machine_time: Dict[int, int] = defaultdict(int)
    cmd_end:      Dict[str, int] = {}
    rows = []

    for cmd in sorted(commandes, key=lambda c: c.Urgence):
        ops      = ops_by_recette.get(cmd.RecetteId, [])
        prev_end = 0
        for i, op in enumerate(ops):
            key     = (cmd.NumeroCommande, i)
            machine = lns_state.assign.get(key)
            if not machine:
                continue
            nb  = lns_state.nb_lots.get(key, 1)
            ls  = lns_state.lot_size.get(key, op.QuantiteLot)
            dur = op.DureeTotale * nb

            s_pm = max(machine_time[machine.Id], prev_end)
            e_pm = s_pm + dur
            machine_time[machine.Id] = e_pm
            prev_end = e_pm
            cmd_end[cmd.NumeroCommande] = e_pm

            for li in range(nb):
                sp = s_pm + li * op.DureeTotale
                ep = sp   + op.DureeTotale
                sd, _, _ = pm_to_clock(sp)
                ed, _, _ = pm_to_clock(ep)
                rows.append(GanttRow(
                    numeroCommande=cmd.NumeroCommande,
                    quantite=cmd.Quantite,
                    recetteId=cmd.RecetteId,
                    urgence=cmd.Urgence,
                    nomOperation=op.NomOperation,
                    machineId=machine.Id,
                    machineName=machine.NomMachine,
                    startPM=sp, endPM=ep,
                    dureeMinutes=op.DureeMinutes,
                    tempsChargementMinutes=op.TempsChargementMinutes,
                    tempsDecharementMinutes=op.TempsDecharementMinutes,
                    dureeTotale=op.DureeTotale,
                    lotSize=(cmd.Quantite - li * ls if li == nb - 1 else ls),
                    quantiteLot=ls,
                    lotIdx=li, nbLots=nb,
                    dateStart=working_day_date(sd).isoformat(),
                    dateEnd=working_day_date(ed).isoformat(),
                    dateExport=cmd.DateExport,
                ))
    return rows