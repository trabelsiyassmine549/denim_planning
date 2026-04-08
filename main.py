"""
main.py — Entry point for the denim washing production planner.
"""
import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> None:
    print("Denim Washing Production Planner — starting...\n")

    try:
        from solver.cp_sat_solver import solve
        from output.gantt import generate_gantt
    except ModuleNotFoundError as e:
        print(f"Import error: {e}")
        print("Verify that the 'solver/' and 'output/' directories exist and are importable.")
        sys.exit(1)

    results = solve()

    if results:
        output_path = os.path.join(_ROOT, "output", "gantt_chart.html")
        generate_gantt(results, output_path=output_path)
        print(f"\nDone. Open: {output_path}")
    else:
        print("\nNo schedule generated.")


if __name__ == "__main__":
    main()