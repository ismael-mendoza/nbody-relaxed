#!/usr/bin/env python3

from pathlib import Path
import shutil
import argparse
import warnings
import pickle

from relaxed.frames.catalogs import catalog_properties
from relaxed.progenitors import save_progenitors


def setup_paths(args):
    root_path = Path(args.root_path)

    paths = {
        "trees": root_path.joinpath("trees"),
        "progenitor_dir": root_path.joinpath("progenitors"),
        "progenitor_file": root_path.joinpath("progenitors.txt"),
        "summarize_file": root_path.joinpath("summarize.csv"),
    }

    return paths


def write(args, paths):
    assert args.cpus is not None, "Need to specify cpus"
    # Bolshoi
    Mcut = 1e3 * catalog_properties["Bolshoi"][0]

    progenitor_path = paths["progenitor_dir"]

    if progenitor_path.exists() and args.overwrite:
        warnings.warn("Overwriting current progenitor directory")
        shutil.rmtree(progenitor_path)

    progenitor_path.mkdir(exist_ok=False)

    save_progenitors.write_main_line_progenitors(
        paths["trees"], progenitor_path.joinpath("mline"), Mcut, cpus=args.cpus
    )


def merge(paths):
    save_progenitors.merge_progenitors(
        paths["progenitor_dir"], paths["progenitor_file"]
    )


def summarize(paths):
    save_progenitors.summarize_progenitors(
        paths["progenitor_file"], paths["progenitor_table"]
    )


def save_tables(paths, ids_file):
    # save all progenitors into tables in a single h5py file.
    # ids_file is a pickle file with ids that should be saved.
    assert ids_file is not None
    with open(ids_file, "r") as fp:
        ids = pickle.load(fp)
        save_progenitors.save_tables()


def main(args):
    paths = setup_paths(args)

    if args.write:
        write(args, paths)

    elif args.merge:
        merge(paths)

    elif args.summarize:
        summarize(paths)

    elif args.save_tables:
        save_tables(paths, args.ids_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write main line progenitors from tree files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--save-tables", action="store_true")

    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--root-path",
        type=str,
        help="Root path containing all tree associated files for a given catalog.",
        default="/home/imendoza/alcca/nbody-relaxed/data/trees_bolshoi",
    )
    parser.add_argument(
        "--ids-file", type=str, default=None, help="file containing ids to filter."
    )

    pargs = parser.parse_args()

    main(pargs)
