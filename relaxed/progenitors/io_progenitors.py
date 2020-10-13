import multiprocessing
import os
import re
import subprocess
import numpy as np

from astropy.io import ascii
from astropy.table import Table

from . import progenitor_lines

url_skeletons = {
    "Bolshoi": "https://www.slac.stanford.edu/~behroozi/Bolshoi_Trees/tree"
}


def work(task):
    return subprocess.run(task, shell=True)


def download_trees(ncubes, data_dir, url_skeleton):
    """Download all the bolshoi trees from the listed url."""

    if data_dir.exists():
        raise IOError("Directory already exists! Overwriting?")

    downloads_file = data_dir.joinpath("downloads.txt")

    # create file listing all files to be downloaded one-per-line.
    for x in range(0, ncubes):
        for y in range(0, ncubes):
            for z in range(0, ncubes):
                with open(downloads_file, "a") as f:
                    if not os.path.isfile(f"data/trees/tree_{x}_{y}_{z}.dat.gz"):
                        f.write(f"{url_skeleton}_{x}_{y}_{z}.dat.gz\n")

    # then download the files using multiprocessing
    os.chdir(data_dir.as_posix())
    subprocess.run(
        "cat downloads.txt | xargs -n 1 --max-procs 10 --verbose wget", shell=True
    )


def write_main_line_progenitors(tree_dir, out_file_prefix, Mcut, cpus=5):
    """Use the consistent trees package to extract main progenitor lines from downloaded trees."""
    subprocess.run(f"cd {utils.read_tree_path.as_posix()}; make", shell=True)
    cmds = []
    for p in tree_dir.iterdir():
        if p.suffix == ".dat" and p.name.startswith("tree"):
            print(f"Found tree: {p.name}")

            # get numbered part.
            suffx = re.search(r"tree(_\d_\d_\d)\.dat", p.name).groups()[0]
            cmd = (
                f"cd {utils.read_tree_path.as_posix()}; ./read_tree {p.as_posix()} "
                f"{out_file_prefix.as_posix()}{suffx}.txt {Mcut}"
            )
            cmds.append(cmd)

    pool = multiprocessing.Pool(cpus)
    pool.map(work, cmds)


def merge_progenitors(progenitor_dir, progenitor_file):
    """Merge all progenitor files in 'progenitor_dir' into one, save it as 'progenitor_file'.  """
    with open(progenitor_file, "w") as pf:
        for p in progenitor_dir.iterdir():
            assert p.name.startswith("mline")
            print(p.name)
            with open(p, "r") as single_pf:
                # ignore headers, etc. which is accounted for in progenitors.py
                pf.write(single_pf.read())


def save_tables(progenitor_file, output_file, ids_to_save):
    """Save tables of each of the progenitor lines (using prog_generator) to a single `output_file`
    which has a .hdf5 extension. The final IDs extracted are also saved in the same file."""
    assert output_file.suffix == ".hdf5"
    prog_generator = progenitor_lines.get_prog_lines_generator(progenitor_file)

    new_ids = []
    for i, prog in enumerate(prog_generator):
        if prog.root_id in ids_to_save:
            new_ids.append(prog.root_id)
            prog.cat.write(output_file, path=str(prog.root_id), compression=True)

    # save ids as well
    new_ids = np.array(new_ids).reshape(-1, 1)
    new_ids = Table(data=new_ids, names=["id"])
    new_ids.write(output_file, path="id", compression=True)


def create_z_file(z_dir, table_file):
    assert table_file.suffix == ".hdf5"
    ids = Table.read(table_file, path="id")["id"]

    z_map = {}
    m_map = {}
    z_map_file = z_dir.joinpath("z_map.json")
    m_map_file = z_dir.joinpath("m_map.json")
    count_m = 0
    count_z = 0

    for _id in ids:
        prog_cat = Table.read(table_file, path=str(_id))

        assert prog_cat[0]["halo_id"] == _id
        assert (
            max(prog_cat["scale"]) == prog_cat["scale"][0]
        ), "make sure order is correct"

        Mvir = prog_cat[0]["mvir"]

        for row in prog_cat:
            a = row["scale"]
            m = row["mvir"] / Mvir

            if a not in z_map:
                z_map[a] = count_z
                count_z += 1
            if m not in m_map:
                m_map[m] = count
                count_m += 1
            z_file = z_dir.joinpath(f"{z_map[a]}.csv")
            m_file = z_dir.joinpath(f"{m_map[m]}.csv")

            fieldnames = ["root_id", "mvir"]

            if not z_file.exists():
                with open(z_file, "w") as zf:
                    writer = csv.DictWriter(zf, fieldnames)
                    writer.writeheader()
            with open(z_file, "a") as zf:
                writer = csv.DictWriter(zf, fieldnames)
                dct = {"root_id": root_id, "mvir": mvir}
                writer.writerow(dct)
                zf.flush()


def summarize_progenitors(progenitor_file, out_file):
    """Write the summary statistics of all the progenitors in progenitor_file into a
    table with the root id.
    """
    assert out_file.as_posix().endswith(".csv")

    prog_generator = progenitor_lines.get_prog_lines_generator(progenitor_file)
    rows = []
    names = ["id", "a2", "alpha"]
    for prog in prog_generator:
        rows.append((prog.root_id, prog.get_a2(), prog.get_alpha()))

    t = Table(rows=rows, names=names)
    ascii.write(t, out_file, format="csv")
