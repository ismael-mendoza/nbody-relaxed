#!/usr/bin/env python3

from pathlib import Path
import shutil
import argparse
import warnings
import pickle
import click

from relaxed.frames.catalogs import catalog_properties
from relaxed.progenitors import io_progenitors


def get_json_dict(json_file):
    pass


@click.group()
@click.option("--overwrite", default=False)
@click.option("--root", default=Path(__file__).absolute().parent.parent.as_posix())
@click.option("--output-dir", default="output")
@click.option("--catalog-name", default="M11")
@click.option("--minh-catalog", help="Minh catalog file to read")
@click.option("--m-low", default=1e11, help="lower log-mass of halo considered.")
@click.option("--m-high", default=1e12, help="high log-mass of halo considered.")
@click.pass_context
def pipeline(ctx, overwrite, root, output_dir, minh_catalog, m_low, m_high):
    ctx.ensure_object(dict)
    output = Path(root).joinpath("temp", output_dir)
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(exist_ok=False)
    ctx.obj.update(
        dict(
            root=Path(root),
            output=output,
            cat_info=output.joinpath("info.json"),
            id_file=output.joinpath("ids.json"),
            dm_catalog=output.joinpath("dm_cat.csv"),
            minh_catalog=minh_catalog,
            m_low=m_low,
            m_high=m_high,
        )
    )


@pipeline.command()
@click.pass_context
def select_ids(ctx):

    # read given minh catalog in ctx

    # only read 'id' and 'mvir' files

    # need to create appropriate HaloFilter

    # create json file with ids in a list?

    # check upid==-1 no subhaloes should be allowed.

    pass


@pipeline.command()
@click.pass_context
def make_dmcat(ctx):
    # check upid==-1

    # create json file for info and CSV file for actual catalog after reading.
    pass


@pipeline.command()
@click.pass_context
def make_subhaloes(ctx):
    # change function in subhaloes/catalog.py so that it only uses the host IDs to extract info.
    # then use this function here after reading the ID json file.
    subhaloes_name = "subhaloes.csv"


@pipeline.command()
@click.command("--cpus", default=1, help="number of cpus to use.")
@click.command(
    "--trees-dir",
    default="data/trees_bolshoi",
    help="folder containing raw data on all trees.",
)
@click.command(
    "--progenitors-dir",
    default="progenitors",
    help="dir in output containing progenitor info",
)
@click.pass_context
def progenitors(ctx, trees_dir, progenitors_dir):
    trees_dir = ctx["root"].joinpath(trees_dir)
    progenitors_dir = ctx["output"].joinpath(progenitors_dir)
    assert trees_dir.exist()
    progenitors_dir.mkdir(exist_ok=False)

    progenitor_dump = progenitors_dir.joinpath("mline")
    # first write all progenitors to a single file


def write(args, paths):
    """Extract all the progenitor trees in the files contained in paths['trees'], save them to
    progenitor_path with prefix 'mline'."""
    assert args.cpus is not None, "Need to specify cpus"
    # Bolshoi
    Mcut = 1e3 * catalog_properties["Bolshoi"][0]

    progenitor_path = paths["progenitor_dir"]

    if progenitor_path.exists() and args.overwrite:
        warnings.warn("Overwriting current progenitor directory")
        shutil.rmtree(progenitor_path)

    progenitor_path.mkdir(exist_ok=False)

    io_progenitors.write_main_line_progenitors(
        paths["trees"], progenitor_path.joinpath("mline"), Mcut, cpus=args.cpus
    )


def merge(paths):
    io_progenitors.merge_progenitors(paths["progenitor_dir"], paths["progenitor_file"])


def summarize(paths):
    io_progenitors.summarize_progenitors(
        paths["progenitor_file"], paths["summary_file"]
    )


def save_tables(paths, ids_file):

    # save all progenitors into tables in a single h5py file.
    # ids_file is a pickle file with ids that will be saved
    assert ids_file is not None
    with open(ids_file, "r") as fp:
        ids = pickle.load(fp)
        io_progenitors.save_tables()


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
