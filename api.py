"""
api.py  —  FastAPI entry point for the Denim Washing Production Planner
========================================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data.fetcher import load_live_data, validate
from models.schemas import RunRequest, RunResponse
from solver.cp_sat_solver import HAS_ORTOOLS, run_lns, run_cpsat, lns_fallback
from utils.time_utils import PPD, START_DATE, date_to_day_offset

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Denim Planner Optimiser", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "https://localhost:4200"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/planning/health")
async def health():
    return {
        "status":    "ok",
        "ortools":   HAS_ORTOOLS,
        "startDate": START_DATE.isoformat(),
    }


@app.post("/api/planning/run", response_model=RunResponse)
async def run_planning(req: RunRequest):

    # 1. Load live data from .NET
    commandes, machines, ops_by_recette = await load_live_data(
        req.token, req.commandeIds
    )

    if not commandes:
        raise HTTPException(status_code=400, detail="No 'En attente' commandes found")

    machines_ok = [m for m in machines if m.is_available()]
    if not machines_ok:
        raise HTTPException(status_code=400, detail="No functional machines available")

    warnings = validate(commandes, machines, ops_by_recette)

    # 2. LNS — machine assignment heuristic
    lns_state = run_lns(commandes, ops_by_recette, machines_ok)

    # 3a. Fallback if OR-Tools not installed
    if not HAS_ORTOOLS:
        rows = lns_fallback(commandes, ops_by_recette, lns_state)
        makespan_pm = max((r.endPM for r in rows), default=0)

        return RunResponse(
            status="feasible_lns_only",
            makespanDays=makespan_pm // PPD,
            makespanPM=makespan_pm,
            startDate=START_DATE.isoformat(),
            rows=rows,
            warnings=warnings + ["OR-Tools not installed; LNS fallback used"],
        )

    # 3b. CP-SAT — timing optimisation
    max_export_day = max(date_to_day_offset(c.DateExport) for c in commandes)
    horizon        = (max_export_day + 15) * PPD

    # ✅ FIX: use SolverResult instead of unpacking
    result = run_cpsat(
        commandes, ops_by_recette, machines_ok, lns_state, horizon
    )

    makespan_pm = result.makespan_pm

    # ✅ FIX: use result.extract_rows()
    rows = result.extract_rows(commandes, ops_by_recette, lns_state)

    # 4. Response
    return RunResponse(
        status="optimal" if makespan_pm < horizon else "feasible",
        makespanDays=makespan_pm // PPD,
        makespanPM=makespan_pm,
        startDate=START_DATE.isoformat(),
        rows=rows,
        warnings=warnings,
    )