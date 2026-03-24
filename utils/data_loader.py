"""
utils/data_loader.py — Chargement et validation des données JSON
"""
import json
import os
from typing import Tuple, Dict, List

from models.commande import Commande
from models.machine import Machine
from models.recette import Recette, OperationRecette

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_data() -> Tuple[List[Commande], List[Machine], Dict[int, List[OperationRecette]], Dict[int, Recette]]:
    paths = {
        "commandes":  os.path.join(DATA_DIR, "commandes.json"),
        "machines":   os.path.join(DATA_DIR, "machines.json"),
        "recettes":   os.path.join(DATA_DIR, "recettes.json"),
        "operations": os.path.join(DATA_DIR, "operations_recette.json"),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier manquant : {path}")

    with open(paths["commandes"],  encoding="utf-8") as f:
        raw_cmd = json.load(f)["Commandes"]
    with open(paths["machines"],   encoding="utf-8") as f:
        raw_mac = json.load(f)["Machines"]
    with open(paths["recettes"],   encoding="utf-8") as f:
        raw_rec = json.load(f)["Recettes"]
    with open(paths["operations"], encoding="utf-8") as f:
        raw_ops = json.load(f)["OperationsRecette"]

    commandes = [Commande(**c) for c in raw_cmd]
    machines  = [Machine(**m) for m in raw_mac]

    recettes_by_id: Dict[int, Recette] = {r["Id"]: Recette(**r) for r in raw_rec}

    ops_by_recette: Dict[int, List[OperationRecette]] = {}
    for op in raw_ops:
        # Provide defaults for backward compat if fields are missing
        op.setdefault("TempsChargementMinutes",  5)
        op.setdefault("TempsDecharementMinutes", 5)
        o = OperationRecette(**op)
        ops_by_recette.setdefault(o.RecetteId, []).append(o)

    for rec_id in ops_by_recette:
        ops_by_recette[rec_id].sort(key=lambda x: x.Ordre)

    return commandes, machines, ops_by_recette, recettes_by_id


def validate_data(
    commandes: List[Commande],
    machines: List[Machine],
    ops_by_recette: Dict[int, List[OperationRecette]],
    recettes_by_id: Dict,
) -> List[str]:
    warnings = []
    machines_ok   = [m for m in machines if m.is_available()]
    ops_available = {op for m in machines_ok for op in m.operations_list()}

    for cmd in commandes:
        ops = ops_by_recette.get(cmd.RecetteId)
        if not ops:
            warnings.append(
                f"[{cmd.NumeroCommande}] RecetteId={cmd.RecetteId} sans opérations définies"
            )
            continue
        for op in ops:
            if op.NomOperation not in ops_available:
                warnings.append(
                    f"[{cmd.NumeroCommande}] Opération '{op.NomOperation}' "
                    f"sans machine fonctionnelle disponible"
                )
    return warnings