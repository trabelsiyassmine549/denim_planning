"""
data/fetcher.py — Live data fetching from the .NET backend + validation
=======================================================================
Fetches commandes, machines, and recettes/operations via REST,
then builds the internal domain objects used by the solver.
"""

from typing import Dict, List, Optional

import httpx
from fastapi import HTTPException

from models.domain import Commande, Machine, OperationRecette

DOTNET_BASE_URL = "https://localhost:7228/api"   # change port if needed


# ---------------------------------------------------------------------------
# Low-level HTTP helper
# ---------------------------------------------------------------------------

async def _fetch(client: httpx.AsyncClient, path: str, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    r = await client.get(
        f"{DOTNET_BASE_URL}{path}",
        headers=headers,
        timeout=30,
    )
    if r.status_code != 200:
        raise HTTPException(
            status_code=r.status_code,
            detail=f"Backend error on {path}: {r.text[:200]}",
        )
    return r.json()


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

async def load_live_data(
    token: str,
    commande_ids: Optional[List[int]] = None,
):
    """
    Fetch commandes, machines, and recette operations from the .NET backend.

    Returns
    -------
    commandes       : List[Commande]
    machines        : List[Machine]
    ops_by_recette  : Dict[int, List[OperationRecette]]   (sorted by Ordre)
    """
    async with httpx.AsyncClient(verify=False) as client:
        raw_cmds     = await _fetch(client, "/Commandes",  token)
        raw_machines = await _fetch(client, "/Machines",   token)
        raw_recettes = await _fetch(client, "/Recettes",   token)

    # --- Commandes -----------------------------------------------------------
    commandes = [
        Commande(c) for c in raw_cmds
        if c.get("statut", "").lower() == "en attente"
    ]
    if commande_ids:
        commandes = [c for c in commandes if c.Id in commande_ids]

    # --- Machines ------------------------------------------------------------
    machines = [Machine(m) for m in raw_machines]

    # --- Operations (embedded in recettes) -----------------------------------
    ops_by_recette: Dict[int, List[OperationRecette]] = {}
    for r in raw_recettes:
        rid = r["id"]
        ops = [
            OperationRecette({**op, "recetteId": rid})
            for op in r.get("operations", [])
        ]
        ops.sort(key=lambda o: o.Ordre)
        ops_by_recette[rid] = ops

    return commandes, machines, ops_by_recette


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    commandes: List[Commande],
    machines:  List[Machine],
    ops_by_recette: Dict[int, List[OperationRecette]],
) -> List[str]:
    """
    Return a list of human-readable warning strings for data issues
    (missing operations, no capable machine, etc.).
    """
    machines_ok = [m for m in machines if m.is_available()]
    available   = {
        op_name.lower()
        for m in machines_ok
        for op_name in m.operations_list()
    }

    warnings: List[str] = []
    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId)
        if not ops:
            warnings.append(
                f"[{cmd.NumeroCommande}] RecetteId={cmd.RecetteId} has no operations"
            )
            continue
        for op in ops:
            if op.NomOperation.lower() not in available:
                warnings.append(
                    f"[{cmd.NumeroCommande}] Operation '{op.NomOperation}'"
                    f" has no available machine"
                )
    return warnings