import subprocess
import multiprocessing
import re
import os
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
    subprocess.run("cat downloads.txt | xargs -n 1 --max-procs 10 --verbose wget", shell=True)


def write_main_line_progenitors(tree_dir, out_file, cpus=5):
    """
    Use the consistent trees package to extract main progenitor lines from downloaded trees.
    """
    subprocess.run(f"cd {const.read_tree_path.as_posix()}; make", shell=True)
    cmds = []
    for p in tree_dir.iterdir():
        if p.suffix == '.dat' and p.name.startswith('tree'):
            suffx = re.search(r"tree(_\d_\d_\d)\.dat", p.name).groups()[0]
            cmd = f"cd {const.read_tree_path.as_posix()}; ./read_tree {p.as_posix()} " \
                  f"{out_file.as_posix()}{suffx}.txt"
            cmds.append(cmd)

    pool = multiprocessing.Pool(cpus)
    pool.map(work, cmds)
