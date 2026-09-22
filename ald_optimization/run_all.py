"""Run all stages sequentially in the current Python environment."""
import argparse
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument("--quick",action="store_true")
parser.add_argument("--site-area",choices=["angstrom24","table"],default="angstrom24")
args=parser.parse_args()
root=Path(__file__).resolve().parent
scripts=["run_01_analytic.py","run_02_ideal_2d.py","run_03_validation.py","run_04_flow.py",
         "run_05_nonideal.py","run_06_virtual_sensors.py","run_07_3d.py"]
for script in scripts:
    command=[sys.executable,str(root/script),"--site-area",args.site_area]
    if args.quick:command.append("--quick")
    subprocess.run(command,cwd=root,check=True)
