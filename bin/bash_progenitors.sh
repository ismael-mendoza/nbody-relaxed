NAME="bolshoi_trees"
JOBNAME="job_$NAME"
CPUS=10
./bin/remote.py --cmd "./bin/get_progenitor_file.py --cpus $CPUS" --jobname $JOBNAME --time 10:00 --cpus-per-task $CPUS --mem-per-cpu