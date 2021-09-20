#!/usr/bin/env python3
import json
import os
from pathlib import Path
from shutil import copyfile

import click
import numpy as np
from astropy import table
from astropy.io import ascii
from tqdm import tqdm

from relaxed import halo_filters
from relaxed.halo_catalogs import HaloCatalog
from relaxed.halo_catalogs import sims
from relaxed.progenitors.progenitor_lines import get_next_progenitor
from relaxed.subhaloes.catalog import create_subhalo_cat

the_root = Path(__file__).absolute().parent.parent
bolshoi_minh = "Bolshoi/minh/hlist_1.00035.minh"
catname_map = {
    "Bolshoi": "bolshoi",
    "BolshoiP": "bolshoi_p",
}


@click.group()
@click.option("--root", default=the_root.as_posix(), type=str, show_default=True)
@click.option("--outdir", type=str, required=True, help="wrt output")
@click.option("--minh-file", help="./data", type=str, default=bolshoi_minh, show_default=True)
@click.option("--catalog-name", default="Bolshoi", type=str, show_default=True)
@click.option(
    "--all-minh-files", default="bolshoi_catalogs_minh", type=str, show_default=True, help="./data"
)
@click.pass_context
def pipeline(ctx, root, outdir, minh_file, catalog_name, all_minh_files):
    catname = catname_map[catalog_name]

    ctx.ensure_object(dict)
    output = Path(root).joinpath("output", outdir)
    ids_file = output.joinpath("ids.json")
    exist_ok = True if ids_file.exists() else False
    output.mkdir(exist_ok=exist_ok, parents=False)
    data = Path(root).joinpath("data")
    minh_file = data.joinpath(minh_file)

    progenitor_file = Path(root).joinpath("output", f"{catname}_progenitors.txt")
    lookup_file = Path(root).joinpath("output", f"lookup_{catname}.json")
    z_map_file_global = Path(root).joinpath(f"output/{catname}_z_map.json")
    z_map_file = output.joinpath("z_map.json").exists()

    # write z_map file to output if not already there.
    if not z_map_file.exists():
        copyfile(z_map_file_global, z_map_file)

    ctx.obj.update(
        dict(
            root=Path(root),
            data=data,
            output=output,
            catalog_name=catalog_name,
            minh_file=minh_file,
            ids_file=ids_file,
            dm_file=output.joinpath("dm_cat.csv"),
            progenitor_file=progenitor_file,
            lookup_file=lookup_file,
            progenitor_table_file=output.joinpath("progenitor_table.csv"),
            subhalo_file=output.joinpath("subhaloes.csv"),
            all_minh=data.joinpath(all_minh_files),
            lookup_index=output.joinpath("lookup.csv"),
            z_map=z_map_file,
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
        "pid": lambda x: x == -1,
    }
    hfilter = halo_filters.HaloFilter(the_filters, name=ctx.obj["catalog_name"])

    # we only need the params that appear in the filter. (including 'id' and 'mvir')
    minh_params = ["id", "mvir", "pid"]

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
    assert np.all(hcat.cat["pid"] == -1)

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
    assert np.all(hcat.cat["pid"] == -1)

    # save as CSV to be loaded later.
    hcat.save_cat(ctx.obj["dm_file"])


@pipeline.command()
@click.pass_context
def make_progenitors(ctx):
    # total in progenitor_file ~ 382477
    # takes like 2 hrs to run.
    progenitor_file = ctx.obj["progenitor_file"]
    lookup_file = ctx.obj["lookup_file"]
    assert progenitor_file.exists()
    assert lookup_file.exists()
    with open(ctx.obj["ids_file"], "r") as fp:
        ids = np.array(json.load(fp)).astype(int)

    with open(lookup_file, "r") as jp:
        lookup = json.load(jp)
        lookup = {int(k): int(v) for k, v in lookup.items()}

    z_map_file = ctx.obj["z_map"]
    assert z_map_file.exists()

    # first collect all scales from existing z_map
    with open(z_map_file, "r") as fp:
        z_map = dict(json.load(fp))
        z_map = {int(k): float(v) for k, v in z_map.items()}

    # first obtain all scales available + save lines that we want to use.
    prog_lines = []

    # iterate through the progenitor generator, obtaining the haloes that match IDs
    # as well as all available scales (will be nan's if not available for a given line)
    with open(progenitor_file, "r") as pf:
        for id in tqdm(ids, desc="Progress on extracting lines"):
            if id in lookup:  # only extract lines in lookup.
                pos = lookup[id]
                pf.seek(pos, os.SEEK_SET)
                prog_line = get_next_progenitor(pf)
                prog_lines.append(prog_line)

    scales = sorted(list(z_map.values()), reverse=True)

    mvir_names = [f"mvir_a{i}" for i in range(len(scales))]
    # ratio (m2 / m1) where m2 is second most massive co-progenitor.
    cpgr_names = [f"cpgratio_a{i}" for i in range(len(scales))]
    names = ("id", *mvir_names, *cpgr_names)
    values = np.zeros((len(ids), len(names)))
    values[:, 0] = ids
    values[values == 0] = np.nan

    # create an astropy table for a mainline progenitor 'lookup'
    lookup_names = [f"id_a{i}" for i in range(len(scales))]
    lookup_index = np.zeros((len(ids), len(scales)))
    lookup_index[:, 0] = ids
    lookup_index[lookup_index == 0] = -1  # np.nan forces us to use floats when saving.

    for prog_line in prog_lines:
        idx = np.where(ids == prog_line.root_id)[0].item()  # where should I insert this line?
        n_scales = min(len(scales), len(prog_line.cat))
        for s in range(n_scales):
            assert scales[s] == prog_line.cat["scale"][s], "Progenitor was skipped!?"
            mvir = prog_line.cat["mvir"][s]
            values[idx, 1 + s] = mvir
            cpg_mvir = prog_line.cat["coprog_mvir"][s]
            cpg_mvir = 0 if cpg_mvir < 0 else cpg_mvir  # missing values with -1 -> 0
            values[idx, 1 + len(scales) + s] = cpg_mvir / mvir
            lookup_index[idx, s] = prog_line.cat["halo_id"][s]

    prog_table = table.Table(names=names, data=values)
    prog_table.sort("id")
    prog_table["id"] = prog_table["id"].astype(int)
    lookup_index = table.Table(names=lookup_names, data=lookup_index.astype(int))
    lookup_index.sort("id_a0")

    # save final table and json file mapping index to scale
    ascii.write(prog_table, ctx.obj["progenitor_table_file"], format="csv")
    ascii.write(lookup_index, ctx.obj["lookup_index"], format="csv")


@pipeline.command()
@click.option(
    "--threshold",
    default=1.0 / 1000,
    type=float,
    help="Subhalo mass threshold fraction relative to snapshot host halo mass",
    show_default=True,
)
@click.pass_context
def make_subhaloes(ctx, threshold):
    # contains info for subhaloes at all snapshots (including present)
    outfile = ctx.obj["subhalo_file"]
    log_file = ctx.obj["output"].joinpath("subhalo_log.txt")
    all_minh = Path(ctx.obj["all_minh"])
    z_map_file = ctx.obj["z_map"]

    assert all_minh.exists()
    assert z_map_file.exists()

    # get host ids
    with open(ctx.obj["ids_file"], "r") as fp:
        sample_ids = np.array(json.load(fp)).astype(int)

    # first collect all scales from existing z_map
    with open(z_map_file, "r") as fp:
        z_map = dict(json.load(fp))
        z_map = {int(k): float(v) for k, v in z_map.items()}
    z_map_inv = {v: k for k, v in z_map.items()}

    # load lookup index
    lookup_index = ascii.read(ctx.obj["lookup_index"], format="csv")

    f_sub_names = [f"f_sub_a{i}" for i in z_map]
    m2_names = [f"m2_a{i}" for i in z_map]
    table_names = ["id", *f_sub_names, *m2_names]
    data = np.zeros((len(sample_ids), 1 + len(z_map) * 2))
    data[:, 0] = sample_ids
    data[data == 0] = np.nan

    fcat = table.Table(data=data, names=table_names)

    assert np.all(fcat["id"] == lookup_index["id_a0"])

    # check all scales from files are in z_map and viceversa (it's easier to handle this way)
    minh_scales = set()
    for minh_file in all_minh.iterdir():
        if minh_file.suffix == ".minh":
            fname = minh_file.stem
            scale = float(fname.replace("hlist_", ""))
            minh_scales.add(scale)
    assert len(minh_scales.intersection(z_map_inv.keys())) == len(z_map_inv), "Inconsistent scales"

    for minh_file in tqdm(all_minh.iterdir(), total=len(z_map), desc="Progress on .minh files"):
        if minh_file.suffix == ".minh":
            fname = minh_file.stem
            scale = float(fname.replace("hlist_", ""))
            scale_idx = z_map_inv[scale]

            # from the look up index we have the ids at the snapshot that we need to extract.
            host_ids = lookup_index[f"id_a{scale_idx}"]
            keep = host_ids > 0  # remove -1s

            # extract subhalo information for each halo in `ids`.
            subcat = create_subhalo_cat(
                host_ids[keep], minh_file, threshold=threshold, log=log_file
            )
            assert np.all(host_ids[keep] == subcat["id"])
            fcat[f"f_sub_a{scale_idx}"][keep] = subcat["f_sub"]
            fcat[f"m2_a{scale_idx}"][keep] = subcat["m2"]

    ascii.write(fcat, output=outfile, format="csv")


@pipeline.command()
@click.pass_context
def combine_all(ctx):
    # load the 3 catalogs that we will be combining
    dm_cat = ascii.read(ctx.obj["dm_file"], format="csv", fast_reader=True)
    subhalo_cat = ascii.read(ctx.obj["subhalo_file"], format="csv", fast_reader=True)
    progenitor_cat = ascii.read(ctx.obj["progenitor_table_file"], format="csv", fast_reader=True)

    # check all are sorted.
    assert np.array_equal(np.sort(dm_cat["id"]), dm_cat["id"])
    assert np.array_equal(np.sort(subhalo_cat["id"]), subhalo_cat["id"])
    assert np.array_equal(np.sort(progenitor_cat["id"]), progenitor_cat["id"])

    # make sure all 3 catalog have exactly the same IDs.
    assert np.array_equal(dm_cat["id"], subhalo_cat["id"])
    assert np.array_equal(dm_cat["id"], progenitor_cat["id"])

    cat1 = table.join(dm_cat, subhalo_cat, keys=["id"], join_type="inner")
    fcat = table.join(cat1, progenitor_cat, keys=["id"], join_type="inner")

    fcat_file = ctx.obj["output"].joinpath("final_table.csv")

    # save final csv containing all the information.
    ascii.write(fcat, fcat_file, format="csv")


if __name__ == "__main__":
    pipeline()
