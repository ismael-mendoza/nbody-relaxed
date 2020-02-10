"""
    File to call save_one_progenitor for each of the tree files, and save main line progenitors.
"""
from pathlib import Path
import sbatch_utils

tree_path = Path(f"/home/imendoza/alcca/nbody-relaxed/intro/data/trees/")
process_file = "/home/imendoza/alcca/nbody-relaxed/intro/save_one_progenitor.py"
progenitor_path = tree_path.joinpath("progenitors")
jobs_dir = "temp_progenitors"

if not progenitor_path.exists():
    progenitor_path.mkdir()

tree_files = [tree_path.joinpath(f"tree_{x}_{y}_{z}.dat")
              for x in range(4) for y in range(4) for z in range(4)]

for tree_file in tree_files:
    cmd = f"python {process_file} --tree-file {tree_file.as_posix()}"
    sbatch_utils.run_sbatch_job(cmd, jobs_dir, f"save_main_progs_{tree_file.name.replace('.dat','')}", time="2:00")
    break





