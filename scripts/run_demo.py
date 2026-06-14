# DriftGuard : 7 - DEMO SIMULATION
# Simulates 6 months of production pipeline, one by one. Shows drift detection, retraining, and deployment in real time.
"""
Run : 
    python scripts/run_demo.py

Then open: http://localhost:8501 (dashboard)
"""

import os
import sys
import json
import time
import argparse

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

sys.path.insert(0, SCRIPTS_DIR)

from run_drift_detection import DriftRunner
from decision_engine     import DecisionEngine
from retraining_pipeline import RetrainingPipeline
from ab_testing          import ABTester

DELAY = 1.5   # seconds between steps (for visual effect during demo)


def clear():
    os.system("cls" if os.name=="nt" else "clear")

def banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║          🛡️  DriftGuard — LIVE DEMO SIMULATION          ║
║   Adaptive Drift Detection & Retraining for Finance      ║
╚══════════════════════════════════════════════════════════╝
""")

def divider(char="─", n=60):
    print(char * n)

def status_line(month, drift, retrain, outcome, acc):
    drift_s   = "🔴 DRIFT" if drift else "🟢 STABLE"
    retrain_s = "🔄 YES"  if retrain else "⏸  NO"
    oc_s      = {"DEPLOYED":"✅ DEPLOYED","REJECTED":"❌ REJECTED","NO_ACTION":"🟢 NO ACTION"}.get(outcome,"—")
    print(f"  Month {month}  |  {drift_s:<14}  |  Retrain: {retrain_s}  |  {oc_s}  |  Acc: {acc:.4f}")


def run_demo(delay=DELAY, months=None):
    clear()
    banner()
    print("  Initializing DriftGuard pipeline...")
    print("  Dashboard: http://localhost:8501\n")
    time.sleep(delay)

    drift_runner = DriftRunner()
    print("\n  ✅ Drift detector ready")
    time.sleep(0.5)

    decision_engine = DecisionEngine()
    print("  ✅ Decision engine ready")
    time.sleep(0.5)

    retrain_pipeline = RetrainingPipeline()
    print("  ✅ Retraining pipeline ready (Q-Learning enabled)")
    time.sleep(0.5)

    ab_tester = ABTester()
    print("  ✅ A/B tester ready\n")
    time.sleep(delay)

    months_to_run = months or list(range(1, 7))
    history       = []

    divider("═")
    print("  STARTING SIMULATION — 6 MONTHS OF PRODUCTION DATA")
    divider("═")

    for month in months_to_run:
        print(f"\n\n{'─'*60}")
        print(f"  📅 PROCESSING: Month {month}/6")
        print(f"{'─'*60}")
        time.sleep(delay)

        # 1. Drift Detection
        print(f"\n  [STEP 1] Running drift detection...")
        drift_result = drift_runner.run_for_month(month)
        any_drift    = drift_result.get("any_drift_detected", False)
        drift_type   = drift_result.get("actual_drift_injected", "none")
        max_psi      = drift_result.get("max_psi", 0)

        if any_drift:
            print(f"\n  ⚠️  DRIFT DETECTED")
            print(f"     Type    : {drift_type.replace('_',' ').upper()}")
            print(f"     Max PSI : {max_psi:.4f}")
            print(f"     Features: {drift_result.get('drifted_features_psi') or drift_result.get('drifted_features_ks')}")
        else:
            print(f"\n  ✅ No drift — model stable (Max PSI: {max_psi:.4f})")

        time.sleep(delay)

        # 2. Decision Engine
        print(f"\n  [STEP 2] Decision engine evaluating...")
        dec_result  = decision_engine.run_for_month(month)
        should_retrain = dec_result.get("retrain_decision", False)
        outcome        = dec_result.get("outcome", "NO_ACTION")
        time.sleep(delay)

        # 3. Retraining (if triggered)
        rl_action = "—"
        new_acc   = 0
        if should_retrain:
            print(f"\n  [STEP 3] 🔄 Retraining triggered...")
            print(f"           Q-Learning agent deciding strategy...")
            time.sleep(delay * 0.7)
            retrain_result = retrain_pipeline.run(month)
            rl_action  = retrain_result.get("rl_action", "—")
            new_acc    = retrain_result.get("new_accuracy", 0)
            prev_acc   = retrain_result.get("prev_accuracy", 0)
            reward     = retrain_result.get("reward", 0)
            deployed   = retrain_result.get("deployed", False)

            print(f"\n  ── Q-Learning Decision ────────────────────")
            print(f"     Action   : {rl_action.upper()}")
            print(f"     Prev Acc : {prev_acc:.4f}")
            print(f"     New Acc  : {new_acc:.4f}")
            print(f"     Reward   : {reward:+.4f}")
            time.sleep(delay)

            # 4. A/B Testing
            print(f"\n  [STEP 4] A/B Testing — comparing models...")
            ab_result = ab_tester.run(month)
            ab_dec    = ab_result.get("decision", "—")
            outcome   = ab_dec
            time.sleep(delay)
        else:
            print(f"\n  [STEP 3] No retraining needed — skipping")
            new_acc = drift_result.get("concept_drift",{}).get("accuracy_check",{}).get("current_accuracy", 0.97)

        # Summary for this month
        print(f"\n{'═'*60}")
        print(f"  MONTH {month} COMPLETE")
        print(f"{'═'*60}")
        print(f"  Drift      : {'DETECTED → ' + drift_type.upper().replace('_',' ') if any_drift else 'None'}")
        print(f"  Action     : {rl_action if should_retrain else 'None (stable)'}")
        print(f"  Outcome    : {outcome}")
        print(f"  Model Acc  : {new_acc:.4f}")

        history.append({
            "month":   month,
            "drift":   any_drift,
            "dtype":   drift_type,
            "retrain": should_retrain,
            "action":  rl_action,
            "outcome": outcome,
            "acc":     new_acc
        })

        time.sleep(delay * 1.2)

    # Final Summary
    print(f"\n\n{'═'*60}")
    print(f"  🎯 DEMO COMPLETE — FULL SYSTEM SUMMARY")
    print(f"{'═'*60}")
    print(f"\n  {'Month':<8} {'Drift':<12} {'Retrain':<10} {'Action':<12} {'Outcome':<16} Accuracy")
    print(f"  {'─'*65}")
    for h in history:
        drift_s  = "DETECTED" if h["drift"] else "stable"
        print(f"  {h['month']:<8} {drift_s:<12} {'YES' if h['retrain'] else 'no':<10} "
              f"{h['action']:<12} {h['outcome']:<16} {h['acc']:.4f}")

    print(f"\n{'═'*60}")
    print(f"  Q-Learning Agent — Learned Strategy Table")
    print(f"{'═'*60}")
    q_path = os.path.join(BASE_DIR, "models", "q_table.json")
    if os.path.exists(q_path):
        q_table = json.load(open(q_path))
        for state, actions in q_table.items():
            best = max(actions, key=actions.get)
            print(f"  State: {state}")
            for a, v in actions.items():
                marker = " ← BEST" if a == best else ""
                print(f"    {a:10s}: {v:+.4f}{marker}")
            print()

    print(f"\n  ✅ Open dashboard: http://localhost:8501")
    print(f"  ✅ API docs:       http://localhost:8000/docs")
    print(f"\n  DriftGuard demo complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DriftGuard Demo")
    parser.add_argument("--fast",   action="store_true", help="No delays (for testing)")
    parser.add_argument("--months", type=int, nargs="+", help="Specific months e.g. --months 3 5")
    args = parser.parse_args()

    run_demo(
        delay=0.3 if args.fast else DELAY,
        months=args.months
    )