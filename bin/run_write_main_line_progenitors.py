#!/usr/bin/env python3
from pathlib import Path
from relaxed.progenitors.save_progenitors import write_main_line_progenitors
import shutil
import argparse


def main(args):
    tree_path = Path(f"/home/imendoza/alcca/nbody-relaxed/data/trees_bolshoi")
    progenitor_path = tree_path.joinpath("progenitors")
    if progenitor_path.exists():
        shutil.rmtree(progenitor_path)

    progenitor_path.mkdir(exist_ok=False)

    write_main_line_progenitors(tree_path, progenitor_path.joinpath("mline"), cpus=args.cpus)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write main line progenitors from tree files')
    parser.add_argument('--cpus', type=int, required=True)
    pargs = parser.parse_args()

    main(pargs)
