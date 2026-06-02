from datetime import date, datetime

import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from optimisationEngine.data.fetcher import load_live_data, validate
from optimisationEngine.models.schemas import RunRequest, RunResponse
from optimisationEngine.solver.cp_sat_solver import (
    HAS_ORTOOLS,
    run_lns,
    run_cpsat,
    lns_fallback,
)
from optimisationEngine.utils.time_utils import PPD, date_to_day_offset, working_day_date, pm_to_clock

from chatbot.rag.faiss_index import build_index, rebuild_for_planning, _store as _faiss_store
from chatbot.chat_router import router as chatbot_router

app = FastAPI(title="Denim Planner Optimiser", version="2.0.0")

app.include_router(chatbot_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "https://localhost:4200"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "mistral"


def _shift_rows(rows: list, now_pm: int) -> list:
    shifted = []
    for r in rows:
        s_pm = r.startPM + now_pm
        e_pm = r.endPM + now_pm

        s_day = pm_to_clock(s_pm)[0]
        e_day = pm_to_clock(e_pm)[0]

        shifted.append(r.model_copy(update={
            "startPM":   s_pm,
            "endPM":     e_pm,
            "dateStart": working_day_date(s_day).isoformat(),
            "dateEnd":   working_day_date(e_day).isoformat(),
        }))
    return shifted

# Late warning helper


def _collect_late_warnings(rows: list, start_date) -> list[str]:
    from datetime import date as _date

    today = _date.today()
    worst = {}
    for r in rows:
        cmd = r.numeroCommande
        if cmd not in worst or r.dateEnd > worst[cmd][0]:
            worst[cmd] = (r.dateEnd, r.dateExport)

    late_warnings = []
    for cmd, (date_end_str, date_export_str) in sorted(worst.items()):
        try:
            date_end    = _date.fromisoformat(date_end_str)
            date_export = _date.fromisoformat(date_export_str)
        except ValueError:
            continue

        if date_end > date_export:
            days_late = (date_end - date_export).days
            late_warnings.append(
                f"LATE_WARNING|{cmd}|{date_end_str}|{date_export_str}|{days_late}"
            )
        elif date_export < today:
            days_late = (today - date_export).days
            late_warnings.append(
                f"LATE_WARNING|{cmd}|{date_end_str}|{date_export_str}|{days_late}"
            )

    return late_warnings

# Startup

@app.on_event("startup")
async def startup():

    print("[STARTUP] Pre-loading Mistral...")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream":   False,
                "keep_alive": "10m",
                "options":  {"num_predict": 1},
            })
        print("[STARTUP] Mistral ready.")
    except Exception as e:
        print(f"[STARTUP] LLM warmup skipped: {e}")

    print("[STARTUP] Pre-loading embeddings model...")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            await client.post(f"{OLLAMA_URL}/api/embeddings", json={
                "model":  "nomic-embed-text",
                "prompt": "warmup",
            })
        print("[STARTUP] Embeddings ready.")
    except Exception as e:
        print(f"[STARTUP] Embedding warmup skipped: {e}")

    print("[STARTUP] Building FAISS index...")
    try:
        await build_index()
        print("[STARTUP] FAISS index ready.")
    except Exception as e:
        print(f"[STARTUP] FAISS index failed (chatbot still works via SQL-only): {e}")



@app.get("/api/planning/health")
async def health():
    return {
        "status":      "ok",
        "faissVectors": _faiss_store.index.ntotal if _faiss_store.index is not None else 0,
        "faissReady":  _faiss_store.is_ready(),
    }

# After planning saved → FAISS refresh hook


async def _after_planning_saved(planning_id: int):
    try:
        await rebuild_for_planning(planning_id)
    except Exception as e:
        print(f"[FAISS] Refresh failed for planning {planning_id}: {e}")


# Run planning endpoint

@app.post("/api/planning/run", response_model=RunResponse)
async def run_planning(req: RunRequest):

    if req.startDatetime:
        try:
            _start = datetime.fromisoformat(req.startDatetime)
        except ValueError:
            raise HTTPException(422, "Invalid ISO datetime format")
    else:
        _start = datetime.now()

    now_pm    = _start.hour * 60 + _start.minute
    today_iso = _start.date().isoformat()

    try:
        commandes, machines, ops_by_recette = await load_live_data(
            req.token, req.commandeIds
        )
    except Exception as e:
        raise HTTPException(500, f"Data loading error: {str(e)}")

    if not commandes:
        raise HTTPException(
            status_code=400,
            detail={
                "code":    "NO_COMMANDES",
                "message": (
                    "Aucune commande en attente à planifier. "
                    "Vérifiez qu'il existe des commandes avec le statut 'En attente'."
                ),
            },
        )

    machines_ok = [m for m in machines if m.is_available()]

    # ── Guard: no machines available 
    if not machines_ok:
        raise HTTPException(
            status_code=400,
            detail={
                "code":    "NO_MACHINES",
                "message": (
                    "Aucune machine disponible. "
                    "Vérifiez que des machines sont actives et disponibles."
                ),
            },
        )

    warnings = validate(commandes, machines, ops_by_recette)

    lns_state = run_lns(
        commandes, ops_by_recette, machines_ok,
        max_machines_per_op=max(1, min(3, req.maxMachinesPerOp))
    )


    if not HAS_ORTOOLS:
        rows        = lns_fallback(commandes, ops_by_recette, lns_state)
        makespan_pm = max((r.endPM for r in rows), default=0)
        rows        = _shift_rows(rows, now_pm)
        late_warnings = _collect_late_warnings(rows, _start.date())

        return RunResponse(
            status="feasible_lns_only",
            makespanDays=makespan_pm // PPD,
            makespanPM=makespan_pm,
            startDate=today_iso,
            rows=rows,
            warnings=warnings + late_warnings + ["OR-Tools missing"],
        )

    result = run_cpsat(
        commandes, ops_by_recette, machines_ok, lns_state,
        horizon=(max(date_to_day_offset(c.DateExport) for c in commandes) + 15) * PPD
    )

    rows          = result.extract_rows(commandes, ops_by_recette, lns_state)
    rows          = _shift_rows(rows, now_pm)
    late_warnings = _collect_late_warnings(rows, _start.date())

    return RunResponse(
        status="optimal",
        makespanDays=result.makespan_pm // PPD,
        makespanPM=result.makespan_pm,
        startDate=today_iso,
        rows=rows,
        warnings=warnings + late_warnings,
    )