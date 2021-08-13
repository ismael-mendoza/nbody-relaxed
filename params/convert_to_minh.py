from subprocess import call, check_call
import sys
import os
import os.path as path
import glob

def get_catalogues(dir):
    halo_dir = "%s/haloes/raw" % dir
    config_dir = "%s/config" % dir
    hlists = sorted(os.listdir(halo_dir))
    
    restart = "%s/restart.txt" % config_dir
    if path.isfile(restart):
        with open(restart) as fp: target = fp.read().strip()
        if target == "done": return []
        i = hlists.index(target)
        if i == -1:
            raise ValueError("target '%s' in %s is not a valid hlist." % 
                             target, restart)
        hlists = hlists[i:]

    return hlists

def main():
    sim_globs = sys.argv[1:]
    sim_globs = [glob.glob("../simulations/%s" % g) for g in sim_globs]
    print(sim_globs)
    sims = []
    for g in sim_globs: sims += g

    for sim in sims:
        hlists = get_catalogues(sim)
        for hlist in hlists:
            check_call("echo '%s' > %s/config/restart.txt" % (hlist, sim), shell=True)
            check_call("./text_to_minh %s/config/minh.config all_vars.txt %s/haloes/raw/%s %s/haloes/minh_full" % (sim, sim, hlist, sim), shell=True)
            
        call("echo 'done' > %s/config/restart.txt" % sim, shell=True)

if __name__ == "__main__": main()
