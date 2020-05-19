import multiprocessing
import os
import re
import subprocess

from astropy.io import ascii
from astropy.table import Table

from . import progenitors
from ..utils import const

url_skeletons = {
    'Bolshoi': "https://www.slac.stanford.edu/~behroozi/Bolshoi_Trees/tree"
}


def work(task):
    return subprocess.run(task, shell=True)


def download_trees(ncubes, dir_name, url_skeleton):
    """
    Download all the bolshoi trees from the listed url.
    """
    dir_path = const.data_path.joinpath(dir_name)

    if dir_path.exists():
        raise IOError("Directory already exists! Overwriting?")

    downloads_file = dir_path.joinpath("downloads.txt")

    # create file with all files to be downloaded...
    for x in range(0, ncubes):
        for y in range(0, ncubes):
            for z in range(0, ncubes):
                with open(downloads_file, 'a') as f:
                    if not os.path.isfile(f"data/trees/tree_{x}_{y}_{z}.dat.gz"):
                        f.write(f"{url_skeleton}_{x}_{y}_{z}.dat.gz\n")

    # then download the files using multiprocessing
    os.chdir(dir_path.as_posix())
    subprocess.run("cat downloads.txt | xargs -n 1 --max-procs 10 --verbose wget",
                   shell=True)


def write_main_line_progenitors(tree_dir, out_file_prefix, Mcut, cpus=5):
    """
    Use the consistent trees package to extract main progenitor lines from downloaded trees.
    """
    subprocess.run(f"cd {const.read_tree_path.as_posix()}; make", shell=True)
    cmds = []
    for p in tree_dir.iterdir():
        if p.suffix == '.dat' and p.name.startswith('tree'):
            print(f"Found tree: {p.name}")
            suffx = re.search(r"tree(_\d_\d_\d)\.dat", p.name).groups()[0]
            cmd = f"cd {const.read_tree_path.as_posix()}; ./read_tree {p.as_posix()} " \
                  f"{out_file_prefix.as_posix()}{suffx}.txt {Mcut}"
            cmds.append(cmd)

    pool = multiprocessing.Pool(cpus)
    pool.map(work, cmds)


def merge_progenitors(tree_dir, progenitor_dir):
    """
    Merge all progenitor files into one, put it in tree_dir with name "progenitors.txt"
    """
    progenitor_file = tree_dir.joinpath("progenitors.txt")
    with open(progenitor_file, 'w') as pf:
        for p in progenitor_dir.iterdir():
            assert p.name.startswith("mline")
            print(p.name)
            with open(p, 'r') as single_pf:
                pf.write(
                    single_pf.read())  # ignore headers, etc. which is account for in progenitors.py


def summarize_progenitors(progenitor_file, out_file):
    """
    Write the summary statistics of all the progenitors in progenitor_dir into a table with the root id.
    """
    assert out_file.as_posix().endswith(".csv")

    prog_generator = progenitors.get_prog_lines_generator(progenitor_file)
    rows = []
    names = ['id', 'a2', 'alpha']
    for prog in prog_generator:
        rows.append(
            (prog.root_id, prog.get_a2(), prog.get_alpha())
        )

    t = Table(rows=rows, names=names)

    ascii.write(t, out_file, format='csv', fast_writer=True)
