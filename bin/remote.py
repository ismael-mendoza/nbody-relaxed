#!/usr/bin/env python3

"""
Useful functions for submitting sbatch jobs.
"""
import subprocess
import numpy as np
import argparse
from pathlib import Path


# ToDo: Implement iterations, make it clear
def run_sbatch_job(
    cmd,
    job_dir_name,
    job_name,
    time="01:00",
    nodes=1,
    ntasks=1,
    cpus_per_task=1,
    mem_per_cpu="1GB",
    iterations=1,
):
    # prepare files and directories
    jobseed = np.random.randint(1e7)
    jobfile_name = f"{job_name}_{jobseed}.sbatch"
    job_dir = Path(job_dir_name)
    jobfile = job_dir.joinpath(jobfile_name)
    if not job_dir.exists():
        job_dir.mkdir(exist_ok=True)

    with open(jobfile, "w") as f:
        f.writelines(
            "#!/bin/bash\n\n"
            f"#SBATCH --job-name={job_name}\n"
            f"#SBATCH --output={job_dir_name}/%j.out\n"
            f"#SBATCH --time={time}:00\n"
            f"#SBATCH --nodes={nodes}\n"
            f"#SBATCH --ntasks={ntasks}\n"
            f"#SBATCH --cpus-per-task={cpus_per_task}\n"
            f"#SBATCH --mem-per-cpu={mem_per_cpu}\n"
            f"#SBATCH --mail-type=END,FAIL\n"
            f"#SBATCH --mail-user=imendoza@umich.edu\n"
            f"#SBATCH --account=cavestru1\n"
            f"#SBATCH --partition=standard\n"
            f"{cmd}\n"
        )

        # f.writelines(f"#SBATCH --array=1-{iterations}%1000\n")

    subprocess.run(f"sbatch {jobfile.as_posix()}", shell=True)


def main(args):
    run_sbatch_job(
        args.cmd,
        args.job_dir,
        args.job_name,
        args.time,
        args.nodes,
        args.ntasks,
        args.cpus_per_task,
        args.mem_per_cpu,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run commands from this /bin/ directory remotely.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cmd", type=str, required=True)
    parser.add_argument("--job-dir", type=str, required=True)
    parser.add_argument("--job-name", type=str, required=True)

    parser.add_argument("--time", type=str, default="01:00", help="In hours.")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument(
        "--ntasks",
        type=int,
        default=1,
        help="How many processes do you want to " "run?",
    )
    parser.add_argument(
        "--cpus-per-task", type=int, default=1, help="How many cpus per process?"
    )
    parser.add_argument("--mem-per-cpu", type=str, default="1GB")

    pargs = parser.parse_args()
    main(pargs)
