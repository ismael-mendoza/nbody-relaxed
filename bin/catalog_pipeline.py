#!/usr/bin/env python3

from pathlib import Path
import json
import click
import numpy as np
from astropy.io import ascii
from astropy.table import Table

from relaxed.progenitors import io_progenitors, progenitor_lines
from relaxed.halo_catalogs import HaloCatalog, all_props
from relaxed import halo_filters
from relaxed.subhaloes.catalog import create_subhalo_cat

the_root = Path(__file__).absolute().parent.parent
read_trees_dir = the_root.joinpath("packages", "consistent-trees", "read_tree")


@click.group()
@click.option("--root", default=the_root.as_posix(), type=str, show_default=True)
@click.option("--output-dir", default="output", help="w.r.t temp", type=str)
@click.option(
    "--minh-file",
    help="w.r.t. to data",
    type=str,
    default="Bolshoi/minh/hlist_1.00035.minh",
    show_default=True,
)
@click.option("--catalog-name", default="Bolshoi", type=str)
@click.option("--m-low", default=1.5e11, help="lower log-mass of halo considered.")
@click.option("--m-high", default=1e12, help="high log-mass of halo considered.")
@click.option(
    "--num-haloes",
    default=int(1e4),
    type=int,
    help="Desired num haloes in ID file.",
)
@click.pass_context
def pipeline(ctx, root, output_dir, minh_file, catalog_name, m_low, m_high, num_haloes):
    ctx.ensure_object(dict)
    output = Path(root).joinpath("temp", output_dir)
    ids_file = output.joinpath("ids.json")
    exist_ok = True if ids_file.exists() else False
    output.mkdir(exist_ok=exist_ok, parents=False)
    data = Path(root).joinpath("data")
    minh_file = data.joinpath(minh_file)
    ctx.obj.update(
        dict(
            root=Path(root),
            data=data,
            output=output,
            catalog_name=catalog_name,
            minh_file=minh_file,
            ids_file=ids_file,
            dm_file=output.joinpath("dm_cat.csv"),
            subhaloes_file=output.joinpath("subhaloes.csv"),
            progenitor_dir=output.joinpath("progenitors"),
            progenitor_file=output.joinpath("progenitors.txt"),
            m_low=m_low,
            m_high=m_high,
            N=num_haloes,
        )
    )


@pipeline.command()
@click.pass_context
def select_ids(ctx):
    # create appropriate filters
    particle_mass = all_props[ctx.obj["catalog_name"]]["particle_mass"]
    assert ctx.obj["m_low"] > particle_mass * 1e3, f"particle mass: {particle_mass:.3g}"
    the_filters = {
        "mvir": lambda x: (x > ctx.obj["m_low"]) & (x < ctx.obj["m_high"]),
        "upid": lambda x: x == -1,
    }
    hfilter = halo_filters.HaloFilter(the_filters, name=ctx.obj["catalog_name"])

    # we only need the params that appear in the filter. (including 'id' and 'mvir')
    minh_params = [
        "id",
        "mvir",
        "upid",
    ]

    # create catalog
    hcat = HaloCatalog(
        ctx.obj["catalog_name"],
        ctx.obj["minh_file"],
        minh_params,
        hfilter,
        subhalos=False,
    )
    hcat.load_cat_minh()

    # do we have enough haloes?
    # keep only N of them
    assert len(hcat) >= ctx.obj["N"]
    keep = np.random.choice(np.arange(len(hcat)), size=ctx.obj["N"], replace=False)
    hcat.cat = hcat.cat[keep]

    # double check only host haloes are allowed.
    assert np.all(hcat.cat["upid"] == -1)

    # extract ids into a json file, first convert to int's.
    ids = sorted([int(x) for x in hcat.cat["id"]])
    assert len(ids) == ctx.obj["N"]
    with open(ctx.obj["ids_file"], "w") as fp:
        json.dump(ids, fp)


@pipeline.command()
@click.pass_context
def make_dmcat(ctx):
    with open(ctx.obj["ids_file"], "r") as fp:
        ids = np.array(json.load(fp))

    assert np.all(np.sort(ids) == ids)

    id_filter = halo_filters.get_id_filter(ids)
    hfilter = halo_filters.HaloFilter(id_filter)

    # create hcat to store these ids
    # NOTE: Use default halo parameters defined in HaloCatalog.
    hcat = HaloCatalog(ctx.obj["catalog_name"], ctx.obj["minh_file"], hfilter=hfilter)

    # now load using minh to obtain dm catalog
    hcat.load_cat_minh()

    assert np.all(hcat.cat["id"] == ids)
    assert len(hcat) == ctx.obj["N"]
    assert np.all(hcat.cat["upid"] == -1)

    # save as CSV to be loaded later.
    hcat.save_cat(ctx.obj["dm_file"])


@pipeline.command()
@click.pass_context
def make_subhaloes(ctx):
    # change function in subhaloes/catalog.py so that it only uses the host IDs to extract info.
    # then use this function here after reading the ID json file.
    with open(ctx.obj["ids_file"], "r") as fp:
        host_ids = np.array(json.load(fp))
    subcat = create_subhalo_cat(host_ids, ctx.obj["minh_file"])
    ascii.write(subcat, ctx.obj["subhaloes_file"], format="csv")


@pipeline.command()
@click.option("--cpus", help="number of cpus to use.")
@click.option(
    "--trees-dir",
    default="trees_bolshoi",
    help="folder containing raw data on all trees relative to data.",
)
@click.pass_context
def create_progenitor_file(ctx, cpus, trees_dir):
    particle_mass = all_props[ctx.obj["catalog_name"]]["particle_mass"]
    trees_dir = ctx.obj["data"].joinpath(trees_dir)
    progenitor_dir = ctx.obj["progenitor_dir"]
    assert trees_dir.exists()
    progenitor_dir.mkdir(exist_ok=False)
    mcut = particle_mass * 1e3
    prefix = progenitor_dir.joinpath("mline").as_posix()

    # first write all progenitors to multiple files
    io_progenitors.write_main_line_progenitors(
        read_trees_dir, trees_dir, prefix, mcut, cpus
    )

    # then merge all of these into a single file
    io_progenitors.merge_progenitors(progenitor_dir, ctx.obj["progenitor_file"])


@pipeline.command()
@click.pass_context
def create_progenitor_table(ctx):
    # total in progenitor_file ~ 382477
    # takes like 2 hrs to run.
    with open(ctx.obj["ids_file"], "r") as fp:
        ids = set(json.load(fp))

    prog_lines = []
    scales = set()

    # now read the progenitor file using generators.
    # first obtain all scales available, save lines that we want to use.
    prog_generator = progenitor_lines.get_prog_lines_generator(
        ctx.obj["progenitor_file"]
    )

    matches = 0
    for i, prog_line in enumerate(prog_generator):
        if i % 10000 == 0 and i > 0:
            print(i)
            print("matches:", matches)
        if prog_line.root_id in ids:
            if matches == 0:
                # avoid empty set intersection.
                scales = set(prog_line.cat["scale"])
            else:
                scales = scales.intersection(set(prog_line.cat["scale"]))
            prog_lines.append(prog_line)
            matches += 1

    scales = sorted(list(scales), reverse=True)
    z_map = {i: scale for i, scale in enumerate(scales)}
    n_scales = len(scales)
    names = ("id", *[f"mvir_a{i}" for i in range(len(scales))])
    values = np.zeros((len(prog_lines), len(names)))

    for i, prog_line in enumerate(prog_lines):
        values[i, 0] = prog_line.root_id
        values[i, 1:] = prog_line.cat["mvir"][:n_scales]

    t = Table(names=names, data=values)
    progenitor_table_file = ctx.obj["output"].joinpath("progenitor_table.csv")
    ascii.write(t, progenitor_table_file, format="csv")

    z_map_file = ctx.obj["output"].joinpath("z_map.json")

    with open(z_map_file, "w") as fp:
        json.dump(z_map, fp)


@pipeline.command()
@click.pass_context
def combine_all(ctx):
    with open(ctx.obj["ids_file"], "r") as fp:
        ids = set(json.load(fp))

    dm_cat = ascii.read(ctx.obj["dm_file"], format="csv", fast_reader=True)
    subhalo_cat = ascii.read(ctx.obj["subhaloes_file"], format="csv", fast_reader=True)

    assert np.all(dm_cat["m_vir"] == subhalo_cat["mvir"])

    # discard haloes with f_sub > 1


if __name__ == "__main__":
    pipeline()
