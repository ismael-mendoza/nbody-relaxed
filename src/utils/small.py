
import subprocess
import os
from src.utils import const
import gzip
import csv
from pathlib import Path


def download_trees(ncubes, dir_name, url_skeleton="https://www.slac.stanford.edu/~behroozi/Bolshoi_Trees/tree"):
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


def get_stem(filename: str):
    return filename.split('.')[0]


def hlist_dat_to_csv(hlist_file):
    """
    Convert a hlist file to a .csv file.
    """
    hlist_file = Path(hlist_file)

    filename_stem = get_stem(hlist_file.name)
    new_filename = f'{filename_stem}.csv'
    hlist_new_file = hlist_file.parent.joinpath(new_filename)

    with gzip.open(hlist_file, 'rt') as f:
        with open(hlist_new_file, mode='w') as csvfile:
            for i,line in enumerate(f):

                if i % 10000 == 0: # show progress.
                    print(i)

                if i == 0:  # header
                    fieldnames = [name[:name.rfind('(')].strip('#') for name in line.split()]
                    writer = csv.DictWriter(csvfile, fieldnames)
                    writer.writeheader()

                if i >= 58:  # content.
                    dct = {key: value for key,value in zip(fieldnames, line.split())}
                    writer.writerow(dct)

                else:
                    continue  # skip descriptions.
