import subprocess
import os

if os.path.isfile("data/trees/downloads.txt"):
    subprocess.run("rm data/trees/downloads.txt", shell=True)

#create file with all files to be downloaded... 
for x in range(0,5):
    for y in range(0,5):
        for z in range(0,5):
            with open("data/trees/downloads.txt", 'a') as f:
                if not os.path.isfile(f"data/trees/tree_{x}_{y}_{z}.dat.gz"):
                    f.write(f"https://www.slac.stanford.edu/~behroozi/Bolshoi_Trees/tree_{x}_{y}_{z}.dat.gz\n")

#then download the files using multiprocessing
os.chdir("data/trees")
subprocess.run("cat downloads.txt | xargs -n 1 --max-procs 10 --verbose wget", shell=True)
