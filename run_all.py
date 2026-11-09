"""One entry point for everything.

  python run_all.py --synthetic            # E1, E2, E3 (synthetic), E4, E7 ontology -> tables/ figures/ (no API)
  python run_all.py --synthetic --quick    # 1-minute smoke test of the same
  python run_all.py --live-mock            # whole live pipeline, small slice, scripted agent (no API)
  python run_all.py --estimate             # predicted Claude cost of the live plan in config.yaml
  python run_all.py --live --pilot         # tiny real run (~$0.10) to calibrate cost
  python run_all.py --live                 # full live campaign: E0 -> E5 -> E3 -> E2 -> E7 [+E4]
                                           #   (only E0 and E5 call Claude; set agent.profile to
                                           #    'permissive' in config.yaml and run again for the stress profile)
  python run_all.py --report               # rebuild live tables/figures from saved logs
  python run_all.py --all                  # synthetic + live + report
"""
import argparse
import subprocess
import sys


def sh(*args):
    print("\n$ " + " ".join(args), flush=True)
    r = subprocess.run([sys.executable, *args])
    if r.returncode:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("synthetic", "quick", "live_mock", "estimate", "live", "pilot", "report", "all"):
        ap.add_argument("--" + f.replace("_", "-"), action="store_true")
    a = ap.parse_args()
    if not any(vars(a).values()):
        ap.print_help()
        return
    if a.synthetic or a.all:
        sh("-m", "experiments.synthetic_experiments", "--exp", "all", *(["--quick"] if a.quick else []))
        sh("-m", "experiments.ontology_experiments", *(["--quick"] if a.quick else []))
    if a.live_mock:
        sh("-m", "experiments.live_experiments", "--exp", "all", "--mock", "--pilot")
        sh("-m", "experiments.live_report", "--mock")
    if a.estimate:
        sh("-m", "experiments.estimate_cost")
    if a.live or a.all:
        sh("-m", "experiments.live_experiments", "--exp", "all", *(["--pilot"] if a.pilot else []))
    if a.live or a.report or a.all:
        sh("-m", "experiments.live_report")


if __name__ == "__main__":
    main()
