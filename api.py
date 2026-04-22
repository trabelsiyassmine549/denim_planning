"""
api.py  —  FastAPI entry point for the Denim Washing Production Planner
========================================================================
"""

import traceback
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data.fetcher import load_live_data, validate
from models.schemas import RunRequest, RunResponse
from solver.cp_sat_solver import (
    HAS_ORTOOLS,
    run_lns,
    run_cpsat,
    lns_fallback,
    collect_split_warnings,
)
from utils.time_utils import PPD, START_DATE, date_to_day_offset
from chat import router as chat_router, OLLAMA_URL, OLLAMA_MODEL

app = FastAPI(title="Denim Planner Optimiser", version="1.0.0")
app.include_router(chat_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "https://localhost:4200"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)


# ---------------------------------------------------------------------------
# Warmup — load the model into memory at startup so first request is fast
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def warmup_ollama():
    print("[WARMUP] Pre-loading Ollama model into memory...")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model":      OLLAMA_MODEL,
                "messages":   [{"role": "user", "content": "hi"}],
                "stream":     False,
                "keep_alive": "10m",
                "options":    {"num_predict": 1},  # generate just 1 token — fast
            })
        print("[WARMUP] Model ready.")
    except Exception as e:
        print(f"[WARMUP] Could not pre-load model (Ollama may not be running): {e}")


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

    print(f"\n{'='*60}")
    print(f"[API] POST /api/planning/run")
    print(f"[API] commandeIds = {req.commandeIds}")
    print(f"[API] maxMachinesPerOp = {req.maxMachinesPerOp}")
    print(f"{'='*60}")

    try:
        commandes, machines, ops_by_recette = await load_live_data(
            req.token, req.commandeIds
        )
    except Exception as e:
        print(f"[API] ERROR loading data: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Data loading error: {str(e)}")

    print(f"[API] Loaded {len(commandes)} commandes, {len(machines)} machines")

    if not commandes:
        raise HTTPException(status_code=400, detail="No 'En attente' commandes found")

    machines_ok = [m for m in machines if m.is_available()]
    if not machines_ok:
        raise HTTPException(status_code=400, detail="No functional machines available")

    print(f"[API] {len(machines_ok)} machines available")

    warnings = validate(commandes, machines, ops_by_recette)
    if warnings:
        print(f"[API] Validation warnings: {warnings}")

    max_machines_per_op = max(1, min(3, req.maxMachinesPerOp))
    print(f"[API] Using max_machines_per_op = {max_machines_per_op}")

    try:
        lns_state = run_lns(
            commandes, ops_by_recette, machines_ok,
            max_machines_per_op=max_machines_per_op,
        )
    except Exception as e:
        print(f"[API] ERROR in LNS: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"LNS solver error: {str(e)}")

    if max_machines_per_op > 1:
        try:
            split_warnings = collect_split_warnings(
                commandes, ops_by_recette, lns_state, max_machines_per_op
            )
            warnings = warnings + split_warnings
        except Exception as e:
            print(f"[API] WARNING: collect_split_warnings failed: {e}")

    if not HAS_ORTOOLS:
        print("[API] OR-Tools not available, using LNS fallback")
        try:
            rows = lns_fallback(commandes, ops_by_recette, lns_state)
        except Exception as e:
            print(f"[API] ERROR in LNS fallback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"LNS fallback error: {str(e)}")

        makespan_pm = max((r.endPM for r in rows), default=0)

        return RunResponse(
            status="feasible_lns_only",
            makespanDays=makespan_pm // PPD,
            makespanPM=makespan_pm,
            startDate=START_DATE.isoformat(),
            rows=rows,
            warnings=warnings + ["OR-Tools not installed; LNS fallback used"],
        )

    try:
        max_export_day = max(date_to_day_offset(c.DateExport) for c in commandes)
        horizon        = (max_export_day + 15) * PPD
        print(f"[API] CP-SAT horizon = {horizon} PM ({horizon // PPD} days)")

        result = run_cpsat(
            commandes, ops_by_recette, machines_ok, lns_state, horizon
        )
    except RuntimeError as e:
        print(f"[API] CP-SAT RuntimeError: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"[API] ERROR in CP-SAT: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"CP-SAT error: {str(e)}")

    try:
        makespan_pm = result.makespan_pm
        rows = result.extract_rows(commandes, ops_by_recette, lns_state)
    except Exception as e:
        print(f"[API] ERROR extracting results: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Result extraction error: {str(e)}")

    print(f"[API] Done. makespan={makespan_pm} PM, {len(rows)} rows generated")

    return RunResponse(
        status="optimal" if makespan_pm < horizon else "feasible",
        makespanDays=makespan_pm // PPD,
        makespanPM=makespan_pm,
        startDate=START_DATE.isoformat(),
        rows=rows,
        warnings=warnings,
    )