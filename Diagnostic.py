"""
Diagnostic.py — Feasibility analysis for the denim washing production planner.
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


def analyze() -> None:
    print("=" * 80)
    print("   DIAGNOSTIC — DENIM WASHING PRODUCTION PLANNER")
    print("=" * 80)
    print()

    commandes, machines, ops_by_recette, recettes_by_id = load_data()

    warnings = validate_data(commandes, machines, ops_by_recette, recettes_by_id)
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  {w}")
        print()

    machines_ok   = [m for m in machines if m.is_available()]
    machines_down = [m for m in machines if not m.is_available()]

    print(f"MACHINES ({len(machines_ok)}/{len(machines)} functional)")
    print("-" * 80)
    for m in machines_ok:
        print(f"  {m.Id:3d}  {m.NomMachine:15s}  cap={m.CapaciteMax:3d}pcs  {m.Operations}")
    print()

    if machines_down:
        print(f"OUT OF SERVICE ({len(machines_down)})")
        for m in machines_down:
            print(f"  {m.Id}  {m.NomMachine}")
        print()

    # Capacity per operation
    cap_by_op: dict  = {}
    machines_by_op: dict = {}
    for m in machines_ok:
        for op in m.operations_list():
            key = op.lower()
            cap_by_op[key]   = cap_by_op.get(key, 0) + m.CapaciteMax
            machines_by_op.setdefault(key, []).append(m)

    print("CAPACITY BY OPERATION (functional machines only)")
    print("-" * 80)
    for op_key in sorted(cap_by_op):
        nb = len(machines_by_op[op_key])
        print(f"  {op_key:22s}: {nb:2d} machine(s) | cumulative capacity = {cap_by_op[op_key]} pcs/cycle")
    print()

    # Order analysis
    print(f"ORDERS ({len(commandes)} total)")
    print("-" * 80)

    all_ok = True
    for cmd in commandes:
        export_offset = date_to_offset(cmd.DateExport)
        ops           = ops_by_recette.get(cmd.RecetteId, [])
        recette       = recettes_by_id.get(cmd.RecetteId)
        urg_label     = URGENCE_LABELS.get(cmd.Urgence, f"Urgence {cmd.Urgence}")

        print(f"\n  {cmd.NumeroCommande}")
        print(f"  Qty={cmd.Quantite}pcs  Urgence={cmd.Urgence} ({urg_label})  "
              f"Recette={recette.NomRecette if recette else '???'} (Id={cmd.RecetteId})")
        print(f"  Export: {cmd.DateExport} (J+{export_offset})")

        if not ops:
            print(f"  [ERROR] RecetteId={cmd.RecetteId} has no operations defined")
            all_ok = False
            continue

        total_dur = 0
        feasible  = True

        for op in ops:
            nb_cycles  = math.ceil(cmd.Quantite / op.QuantiteLot)
            op_dur     = op.DureeMinutes * nb_cycles
            total_dur += op_dur
            avail      = machines_by_op.get(op.NomOperation.lower(), [])
            status     = "OK" if avail else "ERROR: NO MACHINE"
            print(f"    {op.NomOperation:22s}  {op.DureeMinutes:3d}min x {nb_cycles:2d}  = {op_dur:5d}min  "
                  f"lot={op.QuantiteLot}pcs  {status}")
            if not avail:
                feasible = False
                all_ok   = False

        available_min = export_offset * WORK_MINS_PER_DAY
        print(f"  Estimated total : {total_dur} min")
        print(f"  Available time  : {available_min} min (J0 to J+{export_offset})")

        if available_min <= 0:
            print(f"  [INFEASIBLE] Zero or negative time window")
            all_ok = False
        elif total_dur > available_min:
            print(f"  [INFEASIBLE] Deficit of {total_dur - available_min} min")
            all_ok = False
        elif not feasible:
            print(f"  [INFEASIBLE] One or more operations have no available machine")
        else:
            slack = available_min - total_dur
            print(f"  [FEASIBLE]   Slack = {slack} min ({slack / WORK_MINS_PER_DAY:.1f} day(s))")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if all_ok:
        print("All orders appear feasible with current resources.")
    else:
        print("Some orders have feasibility issues.")
        print("Recommended actions:")
        print("  1. Bring out-of-service machines back online")
        print("  2. Revise export dates for deficit orders")
        print("  3. Reduce quantities or split urgent orders")

    # Load by operation
    print("\n" + "=" * 80)
    print("ESTIMATED LOAD BY OPERATION")
    print("=" * 80)

    total_by_op: dict = {}
    for cmd in commandes:
        for op in ops_by_recette.get(cmd.RecetteId, []):
            nb_cyc = math.ceil(cmd.Quantite / op.QuantiteLot)
            key    = op.NomOperation.lower()
            total_by_op[key] = total_by_op.get(key, 0) + op.DureeMinutes * nb_cyc

    for op_key in sorted(total_by_op):
        total_min  = total_by_op[op_key]
        nb_mach    = len(machines_by_op.get(op_key, []))
        hm_needed  = total_min / 60
        hm_avail   = nb_mach * 8
        days_needed = math.ceil(hm_needed / hm_avail) if hm_avail else 999
        status     = "OK" if days_needed <= 20 else "WARNING: HIGH LOAD"
        print(f"  {op_key:22s}: {total_min:6d} min  {nb_mach} machine(s) x 8h = {hm_avail}h/day")
        print(f"  {'':22s}  {status}  {hm_needed:.1f}h needed -> ~{days_needed} day(s)")
    print()


if __name__ == "__main__":
    analyze()