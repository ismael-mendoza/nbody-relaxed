"""
Useful functions for submitting sbatch jobs.
"""
import subprocess
import numpy as np


# ToDo: Implement iterations, make it clear
def run_sbatch_job(cmd, jobs_dir, jobname, time='01:00',
                   ntasks=1, cpus_per_task=1, mem_per_cpu='1GB', iterations=1):
    # prepare files and directories
    jobseed = np.random.randint(1e7)
    jobfile_name = f"{jobname}_{jobseed}.sbatch"
    jobfile = jobs_dir.joinpath(jobfile_name)
    if not jobs_dir.exists():
        jobs_dir.mkdir(exist_ok=True)

    with open(jobfile, 'w') as f:
        f.writelines("#!/bin/bash\n\n"
                     f"#SBATCH --job-name={jobname}\n"
                     f"#SBATCH --output={jobs_dir}/%j.out\n"
                     f"#SBATCH --error={jobs_dir}/%j.err\n"
                     f"#SBATCH --time={time}:00\n"
                     f"#SBATCH --nodes={ntasks}\n"
                     f"#SBATCH --ntasks=10\n"          # how many process do you need?
                     f"#SBATCH --cpus-per-task={cpus_per_task}\n"    # how many cpus per process?
                     f"#SBATCH --mem-per-cpu={mem_per_cpu}\n"
                     f"#SBATCH --mail-type=END,FAIL\n"
                     f"#SBATCH --mail-user=imendoza@umich.edu\n"
                     f"#SBATCH --account=cavestru1\n"
                     f"#SBATCH --partition=standard"
                     f"{cmd}\n"
                     )

        # f.writelines(f"#SBATCH --array=1-{iterations}%1000\n")

    subprocess.run(f"sbatch {jobfile.as_posix()}", shell=True)
