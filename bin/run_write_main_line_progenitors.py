#!/usr/bin/env python3
from pathlib import Path
from relaxed.utils.small import write_main_line_progenitors
import shutil

tree_path = Path(f"/home/imendoza/alcca/nbody-relaxed/data/trees_bolshoi")
progenitor_path = tree_path.joinpath("progenitors")
if progenitor_path.exists():
    shutil.rmtree(progenitor_path)

progenitor_path.mkdir(exist_ok=False)

write_main_line_progenitors(tree_path, progenitor_path.joinpath("mline"))

