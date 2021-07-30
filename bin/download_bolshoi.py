#!/usr/bin/env python3
from pathlib import Path
import json
import os
import subprocess

z_map_file = Path("/home/imendoza/nbody-relaxed/bin/bolshoi_z_map.json")
data_dir = Path("/home/imendoza/nbody-relaxed/data/bolshoi_catalogs")
with open(z_map_file, "r") as fp:
    z_map = json.load(fp)
skeleton_url = "https://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/hlist_{}.list.gz"


data_dir.mkdir(exist_ok=True, parents=False)
downloads_file = data_dir.joinpath("downloads.txt")

for k, v in z_map.items():
    with open(downloads_file, "a") as f:
        url = skeleton_url.format(v)
        f.write(f"{url}\n")

# then download the files using multiprocessing
os.chdir(data_dir.as_posix())
subprocess.run("cat downloads.txt | xargs -n 1 --max-procs 15 --verbose wget", shell=True)
