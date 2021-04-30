#!/usr/bin/env python3
import json
from pathlib import Path

import click
import numpy as np
from astropy import table
from astropy.io import ascii

from relaxed import halo_filters
from relaxed.halo_catalogs import HaloCatalog
from relaxed.halo_catalogs import sims
from relaxed.progenitors import progenitor_lines
from relaxed.subhaloes.catalog import create_subhalo_cat

the_root = Path(__file__).absolute().parent.parent
read_trees_dir = the_root.joinpath("packages", "consistent-trees", "read_tree")
catname_map = {
    "Bolshoi": "bolshoi",
    "BolshoiP": "bolshoi_p",
}


@click.group()
@click.option("--root", default=the_root.as_posix(), type=str, show_default=True)
@click.option("--outdir", type=str, required=True, help="wrt output")
@click.option(
    "--minh-file",
    help="wrt to data",
    type=str,
    default="Bolshoi/minh/hlist_1.00035.minh",
    show_default=True,
)
@click.option("--catalog-name", default="Bolshoi", type=str, show_default=True)
@click.pass_context
def pipeline(ctx, root, outdir, minh_file, catalog_name):
    ctx.ensure_object(dict)
    output = Path(root).joinpath("output", outdir)
    ids_file = output.joinpath("ids.json")
    exist_ok = True if ids_file.exists() else False
    output.mkdir(exist_ok=exist_ok, parents=False)
    data = Path(root).joinpath("data")
    minh_file = data.joinpath(minh_file)

    catname = catname_map[catalog_name]
    progenitor_file = Path(root).joinpath("output", f"{catname}_progenitors.txt")
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
            progenitor_file=progenitor_file,
            progenitor_table_file=output.joinpath("progenitor_table.csv"),
        )
    )


@pipeline.command()
@click.option(
    "--m-low",
    default=11.15,
    help="lower log-mass of halo considered.",
    show_default=True,
)
@click.option(
    "--m-high",
    default=11.22,
    help="high log-mass of halo considered.",
    show_default=True,
)
@click.option(
    "--n-haloes",
    default=int(1e4),
    type=int,
    help="Desired num haloes in ID file.",
    show_default=True,
)
@click.pass_context
def make_ids(ctx, m_low, m_high, n_haloes):
    # create appropriate filters
    assert not ctx.obj["ids_file"].exists()
    m_low = 10 ** m_low
    m_high = 10 ** m_high
    particle_mass = sims[ctx.obj["catalog_name"]].particle_mass
    assert m_low > particle_mass * 1e3, f"particle mass: {particle_mass:.3g}"
    the_filters = {
        "mvir": lambda x: (x > m_low) & (x < m_high),
        "upid": lambda x: x == -1,
    }
    hfilter = halo_filters.HaloFilter(the_filters, name=ctx.obj["catalog_name"])

    # we only need the params that appear in the filter. (including 'id' and 'mvir')
    minh_params = ["id", "mvir", "upid"]

    # create catalog
    hcat = HaloCatalog(
        ctx.obj["catalog_name"],
        ctx.obj["minh_file"],
        subhalos=False,
    )
    hcat.load_cat_minh(minh_params, hfilter)

    # do we have enough haloes?
    # keep only N of them
    assert len(hcat) >= n_haloes
    keep = np.random.choice(np.arange(len(hcat)), size=n_haloes, replace=False)
    hcat.cat = hcat.cat[keep]

    # double check only host haloes are allowed.
    assert np.all(hcat.cat["upid"] == -1)

    # extract ids into a json file, first convert to int's.
    ids = sorted([int(x) for x in hcat.cat["id"]])
    assert len(ids) == n_haloes
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

    # create hcat to store these ids, then load from minh
    # NOTE: Use default halo parameters defined in HaloCatalog.
    hcat = HaloCatalog(ctx.obj["catalog_name"], ctx.obj["minh_file"])
    hcat.load_cat_minh(hfilter=hfilter)

    assert np.all(hcat.cat["id"] == ids)
    assert len(hcat) == len(ids)
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
    assert np.all(subcat["id"] == host_ids)


@pipeline.command()
@click.pass_context
def make_progenitors(ctx):
    # total in progenitor_file ~ 382477
    # takes like 2 hrs to run.
    assert ctx.obj["progenitor_file"].exists()
    with open(ctx.obj["ids_file"], "r") as fp:
        ids = set(json.load(fp))

    prog_lines = []

    # now read the progenitor file using generators.
    prog_generator = progenitor_lines.get_prog_lines_generator(ctx.obj["progenitor_file"])

    # first obtain all scales available + save lines that we want to use.
    matches = 0
    scales = set()
    logs_file = ctx.obj["output"].joinpath("logs.txt")

    # iterate through the progenitor generator, obtaining the haloes that match IDs
    # as wel as all available scales (will be nan's if not available for a given line)
    with open(logs_file, "w") as fp:
        for i, prog_line in enumerate(prog_generator):
            if i % 10000 == 0:
                print(i, file=fp)
                print("matches:", matches, file=fp, flush=True)
            if prog_line.root_id in ids:
                scales = scales.union(set(prog_line.cat["scale"]))
                prog_lines.append(prog_line)
                matches += 1

    scales = sorted(list(scales), reverse=True)
    z_map = {i: scale for i, scale in enumerate(scales)}

    mvir_names = [f"mvir_a{i}" for i in range(len(scales))]
    # merger ratio (m2 / m1) where m2 is second most massive progenitor.
    mratio_names = [f"mratio_a{i}" for i in range(len(scales))]
    names = ("id", *mvir_names, *mratio_names)
    values = np.zeros((len(prog_lines), len(names)))
    values[values == 0] = np.nan

    for i, prog_line in enumerate(prog_lines):
        n_scales = len(prog_line.cat["mvir"])
        values[i, 0] = prog_line.root_id
        values[i, 1 : n_scales + 1] = prog_line.cat["mvir"]
        m2_vir = np.array(prog_line.cat["coprog_mvirs"])
        m2_vir[m2_vir < 0] = 0  # missing values with -1 -> 0
        values[i, len(mvir_names) + 1 : len(mvir_names) + n_scales + 1] = m2_vir / np.array(
            prog_line.cat["mvir"]
        )

    t = table.Table(names=names, data=values)
    t.sort("id")
    z_map_file = ctx.obj["output"].joinpath("z_map.json")

    # save final table and json file mapping index to scale
    ascii.write(t, ctx.obj["progenitor_table_file"], format="csv")
    with open(z_map_file, "w") as fp:
        json.dump(z_map, fp)


@pipeline.command()
@click.pass_context
def combine_all(ctx):
    # load the 3 catalogs that we will be combining
    dm_cat = ascii.read(ctx.obj["dm_file"], format="csv", fast_reader=True)
    subhalo_cat = ascii.read(ctx.obj["subhaloes_file"], format="csv", fast_reader=True)
    progenitor_cat = ascii.read(ctx.obj["progenitor_table_file"], format="csv", fast_reader=True)
    progenitor_cat.sort("id")

    # check all are sorted.
    assert np.array_equal(np.sort(dm_cat["id"]), dm_cat["id"])
    assert np.array_equal(np.sort(subhalo_cat["id"]), subhalo_cat["id"])
    assert np.array_equal(np.sort(progenitor_cat["id"]), progenitor_cat["id"])

    # make sure all 3 catalog have exactly the same IDs.
    assert np.array_equal(dm_cat["id"], subhalo_cat["id"])
    common_ids = set(dm_cat["id"]).intersection(set(progenitor_cat["id"]))
    common_ids = np.array(sorted(list(common_ids)))
    ids1 = np.array(dm_cat["id"])
    ids2 = np.array(progenitor_cat["id"])
    keep1 = halo_filters.intersect(ids1, common_ids)
    keep2 = halo_filters.intersect(ids2, common_ids)
    dm_cat = dm_cat[keep1]
    subhalo_cat = subhalo_cat[keep1]
    progenitor_cat = progenitor_cat[keep2]

    # one last check on all IDs.
    assert np.array_equal(dm_cat["id"], subhalo_cat["id"])
    assert np.array_equal(dm_cat["id"], progenitor_cat["id"])
    assert np.array_equal(dm_cat["mvir"], subhalo_cat["mvir"])
    assert np.array_equal(dm_cat["mvir"], progenitor_cat["mvir_a0"])

    cat1 = table.join(dm_cat, subhalo_cat, keys=["id", "mvir"], join_type="inner")
    fcat = table.join(cat1, progenitor_cat, keys=["id"], join_type="inner")

    fcat_file = ctx.obj["output"].joinpath("final_table.csv")

    # save final csv containing all the information.
    ascii.write(fcat, fcat_file, format="csv")


if __name__ == "__main__":
    pipeline()
