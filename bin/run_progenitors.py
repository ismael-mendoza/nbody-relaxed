#!/usr/bin/env python3

from pathlib import Path
import shutil
import warnings
import pickle
import click
import numpy as np

from relaxed.progenitors import io_progenitors
from relaxed.halo_catalogs import HaloCatalog, props
from relaxed import halo_filters


def get_json_dict(json_file):
    pass


@click.group()
@click.option("--overwrite", default=False)
@click.option("--root", default=Path(__file__).absolute().parent.parent.as_posix())
@click.option("--output-dir", default="output")
@click.option("--catalog-name", default="Bolshoi")
@click.option("--minh-catalog", help="Minh catalog file to read")
@click.option("--m-low", default=1e11, help="lower log-mass of halo considered.")
@click.option("--m-high", default=1e12, help="high log-mass of halo considered.")
@click.pass_context
def pipeline(
    ctx, overwrite, root, output_dir, catalog_name, minh_catalog, m_low, m_high
):
    ctx.ensure_object(dict)
    output = Path(root).joinpath("temp", output_dir)
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(exist_ok=False)
    ctx.obj.update(
        dict(
            root=Path(root),
            output=output,
            catalog_name=catalog_name,
            cat_info=output.joinpath("info.json"),
            id_file=output.joinpath("ids.json"),
            dm_catalog=output.joinpath("dm_cat.csv"),
            minh_catalog=minh_catalog,
            m_low=m_low,
            m_high=m_high,
        )
    )


@pipeline.command()
@click.option("--N", default=1e4, help="Desired number of haloes in ID file.")
@click.pass_context
def select_ids(ctx, N):
    # read given minh catalog in ctx

    # createa appropriate filters
    particle_mass = props[ctx["catalog_name"]]
    mass_filter = halo_filters.get_bound_filter("mvir", ctx["m_low"], ctx["m_high"])
    default_filters = halo_filters.get_default_filters(particle_mass, subhalos=False)
    the_filters = halo_filters.join_filters(mass_filter, default_filters)
    hfilter = halo_filters.HaloFilter(the_filters, name=ctx["catalog_name"])

    # we only need the params that appear in the filter. (including 'id' and 'mvir')
    minh_params = [
        "id",
        "mvir",
        "upid",
        "spin",
        "q",
        "vrms",
    ]

    # craete catalog
    hcat = HaloCatalog(
        ctx["name"],
        ctx["min_catalog"],
        minh_params,
        hfilter,
        subhalos=False,
        label="all_haloes",
    )
    hcat.load_cat_minh()

    # do we have enough haloes?
    # keep only N of them
    assert len(hcat) >= N
    keep = np.random.choice(np.arange(len(hcat)), size=N, replace=False)
    hcat.cat = hcat.cat[keep]

    # extract ids into a json file

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
    pipeline()
