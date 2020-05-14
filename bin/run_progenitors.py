#!/usr/bin/env python3
from pathlib import Path
import shutil
import argparse
import warnings

from relaxed.frames.catalogs import catalog_properties
from relaxed.progenitors.save_progenitors import write_main_line_progenitors, merge_progenitors


def write(args):
    assert args.cpu is not None, "Need to specify cpus"
    # Bolshoi
    Mcut = 1e3 * catalog_properties['Bolshoi'][0]

    tree_path = Path(f"/home/imendoza/alcca/nbody-relaxed/data/trees_bolshoi")
    progenitor_path = tree_path.joinpath("progenitors")
    if progenitor_path.exists():
        warnings.warn("Overwriting current progenitor directory")
        shutil.rmtree(progenitor_path)

    progenitor_path.mkdir(exist_ok=False)

    write_main_line_progenitors(tree_path, progenitor_path.joinpath("mline"), Mcut, cpus=args.cpus)


def merge(args):
    tree_path = Path(f"/home/imendoza/alcca/nbody-relaxed/data/trees_bolshoi")
    progenitors_path = tree_path.joinpath("progenitors")
    merge_progenitors(tree_path, progenitors_path)


def summarize(args):
    pass


def main(args):
    if args.write:
        write(args)

    elif args.merge:
        merge(args)

    elif args.summarize:
        summarize(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write main line progenitors from tree files')
    parser.add_argument('--cpus', type=int, default=None)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--merge', action='store_true')

    pargs = parser.parse_args()

    main(pargs)
