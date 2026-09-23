"""One command for the contest pipeline. Existing best.pt avoids retraining."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def call(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=ROOT, check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--retrain', action='store_true')
    args = p.parse_args()
    if args.retrain or not (ROOT / 'outputs' / 'best.pt').exists():
        call('train.py', '--epochs', args.epochs)
    call('media.py')
    call('predict.py')
    call('make_report.py')

