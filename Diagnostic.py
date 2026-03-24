"""
Diagnostic.py — Analyse de faisabilité du planning de lavage denim (schéma DB)
"""
import math
import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.data_loader import load_data, validate_data
from utils.time_utils import date_to_offset, WORK_MINS_PER_DAY

URGENCE_LABELS = {1: "Urgent", 2: "Haute", 3: "Normal", 4: "Basse", 5: "Flexible"}


def analyze():
    print("=" * 80)
    print("   DIAGNOSTIC — PLANNING LAVAGE DENIM")
    print("=" * 80)
    print()

    commandes, machines, ops_by_recette, recettes_by_id = load_data()

    warnings = validate_data(commandes, machines, ops_by_recette, recettes_by_id)
    if warnings:
        print("⚠️  Problèmes détectés :")
        for w in warnings:
            print(f"   • {w}")
        print()

    # ── Ressources disponibles ────────────────────────────────────────────────
    machines_ok   = [m for m in machines if m.is_available()]
    machines_down = [m for m in machines if not m.is_available()]

    print(f"🔧 MACHINES DISPONIBLES ({len(machines_ok)}/{len(machines)} fonctionnelles)")
    print("-" * 80)
    for m in machines_ok:
        print(f"  {m.Id:3d}  {m.NomMachine:15s} | cap={m.CapaciteMax:3d} pcs | {m.Operations}")
    print()

    if machines_down:
        print(f"⚠️  MACHINES HORS SERVICE ({len(machines_down)})")
        for m in machines_down:
            print(f"  ❌ {m.Id} — {m.NomMachine}")
        print()

    # Capacité par opération (case-insensitive grouping)
    cap_by_op: dict = {}
    machines_by_op: dict = {}
    for m in machines_ok:
        for op in m.operations_list():
            op_lower = op.lower()
            cap_by_op[op_lower] = cap_by_op.get(op_lower, 0) + m.CapaciteMax
            machines_by_op.setdefault(op_lower, []).append(m)

    print("📊 CAPACITÉ PAR OPÉRATION (machines fonctionnelles)")
    print("-" * 80)
    for op_lower in sorted(cap_by_op):
        nb = len(machines_by_op[op_lower])
        print(f"  {op_lower:22s}: {nb:2d} machine(s) | capacité cumulée = {cap_by_op[op_lower]} pcs/cycle")
    print()

    # ── Analyse commandes ─────────────────────────────────────────────────────
    print(f"📦 ANALYSE DES COMMANDES ({len(commandes)} commandes)")
    print("-" * 80)

    all_ok = True
    for cmd in commandes:
        export_offset = date_to_offset(cmd.DateExport)
        ops           = ops_by_recette.get(cmd.RecetteId, [])
        recette_name  = recettes_by_id.get(cmd.RecetteId, None)
        urg_label     = URGENCE_LABELS.get(cmd.Urgence, f"Urgence {cmd.Urgence}")

        print(f"\n  {cmd.NumeroCommande}")
        print(f"  Quantité: {cmd.Quantite} pcs | Urgence: {cmd.Urgence} ({urg_label}) | "
              f"Recette: {recette_name.NomRecette if recette_name else '???'} (Id={cmd.RecetteId})")
        print(f"  Export: {cmd.DateExport} (J+{export_offset})")

        if not ops:
            print(f"  ❌ RecetteId={cmd.RecetteId} sans opérations définies!")
            all_ok = False
            continue

        total_duree = 0
        feasible    = True

        for op in ops:
            nb_cycles    = math.ceil(cmd.Quantite / op.QuantiteLot)
            op_total_min = op.DureeMinutes * nb_cycles
            total_duree += op_total_min
            mach_avail   = machines_by_op.get(op.NomOperation.lower(), [])
            status       = "✅" if mach_avail else "❌ AUCUNE MACHINE"
            print(f"    • {op.NomOperation:22s} {op.DureeMinutes:3d}min × {nb_cycles} cycle(s)"
                  f" = {op_total_min:4d}min | lot={op.QuantiteLot}pcs | {status}")
            if not mach_avail:
                feasible = False
                all_ok   = False

        available_min = export_offset * WORK_MINS_PER_DAY
        print(f"  Durée totale estimée : {total_duree} min")
        print(f"  Temps disponible     : {available_min} min (J0 → J+{export_offset})")

        if available_min <= 0:
            print(f"  ❌ INFAISABLE : fenêtre temporelle nulle ou négative")
            feasible = False
            all_ok   = False
        elif total_duree > available_min:
            deficit = total_duree - available_min
            print(f"  ❌ INFAISABLE : déficit de {deficit} min")
            feasible = False
            all_ok   = False
        elif not feasible:
            print(f"  ❌ INFAISABLE : opération(s) sans machine")
        else:
            slack = available_min - total_duree
            print(f"  ✅ FAISABLE — marge {slack} min ({slack / WORK_MINS_PER_DAY:.1f} jour(s))")

    # ── Résumé ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)
    if all_ok:
        print("✅ Toutes les commandes semblent faisables avec les ressources actuelles.")
    else:
        print("❌ Certaines commandes présentent des problèmes de faisabilité.")
        print("Actions recommandées :")
        print("  1. Remettre en service les machines hors fonction")
        print("  2. Réviser les dates d'export pour les commandes déficitaires")
        print("  3. Réduire les quantités ou fractionner les commandes urgentes")

    # ── Charge par opération ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CHARGE ESTIMÉE PAR OPÉRATION")
    print("=" * 80)

    total_by_op: dict = {}
    for cmd in commandes:
        for op in ops_by_recette.get(cmd.RecetteId, []):
            nb_cyc = math.ceil(cmd.Quantite / op.QuantiteLot)
            key = op.NomOperation.lower()
            total_by_op[key] = total_by_op.get(key, 0) + op.DureeMinutes * nb_cyc

    for op_lower in sorted(total_by_op):
        total_min   = total_by_op[op_lower]
        nb_mach     = len(machines_by_op.get(op_lower, []))
        hm_needed   = total_min / 60
        hm_avail    = nb_mach * 8
        days_needed = math.ceil(hm_needed / hm_avail) if hm_avail else 999
        sign        = "✅" if days_needed <= 20 else "⚠️ "
        print(f"  {op_lower:22s}: {total_min:5d} min | {nb_mach} machine(s) × 8h = {hm_avail}h/jour")
        print(f"  {'':22s}  {sign} {hm_needed:.1f}h nécessaires → ~{days_needed} jour(s)")

    print()


if __name__ == "__main__":
    analyze()