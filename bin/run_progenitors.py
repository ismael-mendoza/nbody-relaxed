#!/usr/bin/env python3
from pathlib import Path
import shutil
import argparse
import warnings

from relaxed.frames.catalogs import catalog_properties
from relaxed.progenitors import save_progenitors


def setup_paths(args):
    tree_path = Path(args.tree_path)

    paths = {
        'trees': tree_path,
        'progenitor_dir': tree_path.joinpath("progenitors"),
        'progenitor_file': tree_path.joinpath("progenitors.txt"),
        'progenitor_table': tree_path.joinpath("progenitors.csv")
    }

    return paths


def write(args, paths):
    assert args.cpus is not None, "Need to specify cpus"
    # Bolshoi
    Mcut = 1e3 * catalog_properties['Bolshoi'][0]

    progenitor_path = paths['progenitor_dir']

    if progenitor_path.exists() and args.overwrite:
        warnings.warn("Overwriting current progenitor directory")
        shutil.rmtree(progenitor_path)

    progenitor_path.mkdir(exist_ok=False)

    save_progenitors.write_main_line_progenitors(paths['trees'], progenitor_path.joinpath(
        "mline"), Mcut, cpus=args.cpus)


def merge(args, paths):
    save_progenitors.merge_progenitors(paths['trees'], paths['progenitor_dir'],
                                       paths['progenitor_file'])


def summarize(args, paths):
    save_progenitors.summarize_progenitors(paths['progenitor_file'], paths['progenitor_table'])


def main(args):
    paths = setup_paths(args)
    if args.write:
        write(args, paths)

    elif args.merge:
        merge(args, paths)

    elif args.summarize:
        summarize(args, paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write main line progenitors from tree files',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cpus', type=int, default=None)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--overwrite', action='store_true')

    parser.add_argument('--tree-path', type=str, help="Path containing raw tree files",
                        default="/home/imendoza/alcca/nbody-relaxed/data/trees_bolshoi")

    pargs = parser.parse_args()

    main(pargs)
