#!/usr/bin/env python3

from pathlib import Path
import shutil
import json
import click
import numpy as np
from astropy.io import ascii
from astropy.table import Table

from relaxed.progenitors import io_progenitors, progenitor_lines
from relaxed.halo_catalogs import HaloCatalog, all_props
from relaxed import halo_filters
from relaxed.subhaloes.catalog import create_subhalo_cat

default_root = Path(__file__).absolute().parent.parent
read_trees_dir = default_root.joinpath("packages", "consistent_trees", "read_tree")


@click.group()
@click.option("--overwrite", default=False)
@click.option("--root", default=default_root.as_posix())
@click.option("--output-dir", default="output", help="Relative to temp")
@click.option("--minh-file", help="Minh catalog file to read")
@click.option("--catalog-name", default="Bolshoi")
@click.option("--m-low", default=1e11, help="lower log-mass of halo considered.")
@click.option("--m-high", default=1e12, help="high log-mass of halo considered.")
@click.option("--N", default=1e4, help="Desired number of haloes in ID file.")
@click.pass_context
def pipeline(
    ctx, overwrite, root, output_dir, minh_file, catalog_name, m_low, m_high, N
):
    ctx.ensure_object(dict)

    # output directly lives in side temp
    output = Path(root).joinpath("temp", output_dir)
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(exist_ok=False, parents=False)
    ctx.obj.update(
        dict(
            root=Path(root),
            output=output,
            catalog_name=catalog_name,
            ids_file=output.joinpath("ids.json"),
            dm_file=output.joinpath("dm_cat.csv"),
            subhaloes_file=output.joinpath("subhaloes.csv"),
            progenitor_dir=output.joinpath("progenitors"),
            progenitor_file=output.joinpath("progenitors.txt"),
            minh_file=minh_file,
            m_low=m_low,
            m_high=m_high,
            N=N,
        )
    )


@pipeline.command()
@click.pass_context
def select_ids(ctx):

    # create appropriate filters
    particle_mass = all_props[ctx["catalog_name"]]["particle_mass"]
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

    # create catalog
    hcat = HaloCatalog(
        ctx["catalog_name"],
        ctx["minh_file"],
        minh_params,
        hfilter,
        subhalos=False,
    )
    hcat.load_cat_minh()

    # double check only host haloes are allowed.
    assert np.all(hcat.cat["upid"] == -1)

    # do we have enough haloes?
    # keep only N of them
    assert len(hcat) >= ctx["N"]
    keep = np.random.choice(np.arange(len(hcat)), size=ctx["N"], replace=False)
    hcat.cat = hcat.cat[keep]

    # extract ids into a json file
    ids = list(hcat.cat["id"])
    with open(ctx["ids_file"], "w") as fp:
        json.dump(ids, fp)


@pipeline.command()
@click.pass_context
def make_dmcat(ctx):
    # read json file
    with open(ctx["ids_file"], "r") as fp:
        ids = np.array(json.load(fp))

    id_filter = halo_filters.get_id_filter(ids)
    hfilter = halo_filters.HaloFilter(id_filter)

    # create hcat to store these ids (use default halo parameters)
    hcat = HaloCatalog(ctx["catalog_name"], ctx["minh_file"], hfilter=hfilter)

    # now load using minh to obtain dm catalog
    hcat.load_cat_minh()

    # check upid==-1 and length is appropriate
    assert np.all(hcat.cat["upid"] == -1)
    assert len(hcat) == ctx["N"]

    # save as CSV to be loaded later.
    hcat.save_cat(ctx["dm_file"])


@pipeline.command()
@click.pass_context
def make_subhaloes(ctx):
    # change function in subhaloes/catalog.py so that it only uses the host IDs to extract info.
    # then use this function here after reading the ID json file.
    with open(ctx["ids_file"], "r") as fp:
        host_ids = np.array(json.load(fp))
    subcat = create_subhalo_cat(host_ids, ctx["minh_file"])
    ascii.write(subcat, ctx["subhaloes_file"], format="csv")


@pipeline.command()
@click.option("--cpus", help="number of cpus to use.")
@click.option(
    "--trees-dir",
    default="data/trees_bolshoi",
    help="folder containing raw data on all trees.",
)
@click.pass_context
def create_progenitor_file(ctx, cpus, trees_dir):
    trees_dir = ctx["root"].joinpath(trees_dir)
    progenitor_dir = ctx["progenitors"]
    assert trees_dir.exist()
    progenitor_dir.mkdir(exist_ok=False)
    particle_mass = all_props[ctx["catalog_name"]]["particle_mass"]
    mcut = particle_mass * 1e3
    prefix = progenitor_dir.joinpath("mline").as_posix()

    # first write all progenitors to multiple files
    io_progenitors.write_main_line_progenitors(
        read_trees_dir, trees_dir, prefix, mcut, cpus
    )

    # then merge all of these into a single file
    io_progenitors.merge_progenitors(progenitor_dir, ctx["progenitor_file"])


@pipeline.command()
@click.pass_context
def create_progenitor_table(ctx):
    with open(ctx["ids_file"], "r") as fp:
        ids = set(json.load(fp))

    prog_lines = []
    scales = set()

    # now read the progenitor file using generators.
    # first obtain all scales available, save lines that we want to use.
    prog_generator = progenitor_lines.get_prog_lines_generator(ctx["progenitor_file"])
    for prog_line in prog_generator:
        if prog_line.root_id in ids:
            scales = scales.union(set(prog_line.cat["scale"]))
            prog_lines.append(prog_line)

    scales = sorted(list(scales), reverse=True)
    z_map = {i: scales for i in range(len(scales))}
    names = ("id", *[f"m_a_{i}" for i in range(len(scales))])
    n_lines = len(prog_lines)
    values = np.zeros(len(names), n_lines)

    for i, prog_line in enumerate(prog_lines):
        values[0] = prog_line.root_id
        values[1:, i] = prog_line["mvir"]

    t = Table(names=names, data=values)
    ascii.write(t, "progenitor_table.csv", format="csv")

    with open("z_map.json", "w") as fp:
        json.dump(z_map, fp)
