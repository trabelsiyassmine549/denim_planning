"""
main.py — Point d'entrée du système de planification denim
"""
import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    print("Demarrage du systeme de planification denim...\n")

    try:
        from solver.cp_sat_solver import solve
        from output.gantt import generate_gantt
    except ModuleNotFoundError as e:
        print(f"Erreur d'import: {e}")
        print("   Verifiez que les dossiers 'solver/' et 'output/' existent")
        sys.exit(1)

    results = solve()

    if results:
        generate_gantt(results, output_path="output/gantt_chart.html")
        print("\nPlanification terminee. Ouvrez output/gantt_chart.html")
    else:
        print("\nAucun planning genere.")


if __name__ == "__main__":
    main()