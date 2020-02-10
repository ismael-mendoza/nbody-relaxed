"""
Useful functions for submitting sbatch jobs.
"""
import subprocess
from pathlib import Path
import numpy as np

batch_dir = Path("/home/imendoza/alcca/nbody-relaxed/intro/temps/batches")
logs = Path("/home/imendoza/alcca/nbody-relaxed/intro/temps/batches/logs")


def run_sbatch_job(cmd, jobs_dir_name, jobname, time='01:00', memory='5GB', iterations=1):

    # prepare files and directories
    jobseed = np.random.randint(1e6)
    identifier = jobname + "_" + str(jobseed)
    jobfile_name = f"job_{identifier}.sbatch"
    jobs_dir = batch_dir.joinpath(jobs_dir_name)
    jobfile = jobs_dir.joinpath(jobfile_name)

    if not jobs_dir.exists():
        jobs_dir.mkdir()

    with open(jobfile.as_posix(), 'w') as f:
        f.writelines("#!/bin/bash\n")
        f.writelines(f"#SBATCH --job-name={identifier}\n")
        f.writelines(f"#SBATCH --output={logs}/{identifier}.out\n")
        f.writelines(f"#SBATCH --error={logs}/{identifier}.err\n")
        f.writelines(f"#SBATCH --time={time}:00\n")
        f.writelines(f"#SBATCH --mem={memory}\n")
        f.writelines(f"#SBATCH --mail-type=END,FAIL\n")
        f.writelines(f"#SBATCH --mail-user=imendoza@umich.edu\n")
        f.writelines(f"#SBATCH --account=cavestru\n")
        f.writelines(f"{cmd}\n")
        # f.writelines(f"#SBATCH --array=1-{iterations}%1000\n")

    subprocess.run(f"sbatch {jobfile.as_posix()}", shell=True)
