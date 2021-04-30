#!/usr/bin/env python3
from pathlib import Path

import click

from relaxed.halo_catalogs import sims
from relaxed.progenitors import io_progenitors

the_root = Path(__file__).absolute().parent.parent
read_trees_dir = the_root.joinpath("consistent-trees", "read_tree")
catname_map = {
    "Bolshoi": "bolshoi",
    "BolshoiP": "bolshoi_p",
}


@click.command()
@click.option("--root", default=the_root.as_posix(), type=str, show_default=True)
@click.option("--catalog-name", default="Bolshoi", type=str, show_default=True)
@click.option("--cpus", type=int, help="number of cpus to use.")
def make_progenitor_file(root, catalog_name, cpus):
    # setup required names and directories first.
    catname = catname_map[catalog_name]
    prog_name = f"{catname}_progenitors"
    root = Path(root)
    trees_dir = Path(root).joinpath("data", f"trees_{catname}")
    progenitor_dir = root.joinpath("output", prog_name)
    progenitor_file = root.joinpath("output", prog_name).with_suffix(".txt")
    assert trees_dir.exists()
    progenitor_dir.mkdir(exist_ok=True)
    particle_mass = sims[catalog_name].particle_mass
    mcut = particle_mass * 1e3

    # prefix to be used to save the results of each of the tree files.
    prefix = progenitor_dir.joinpath("mline")

    # write all progenitors to multiple files using consistent trees.
    io_progenitors.write_main_line_progenitors(read_trees_dir, trees_dir, prefix, mcut, cpus)

    # then merge all of these into a single file
    io_progenitors.merge_progenitors(progenitor_dir, progenitor_file)


if __name__ == "__main__":
    make_progenitor_file()
