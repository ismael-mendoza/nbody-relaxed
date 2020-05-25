ROOT_DIR="/home/imendoza/alcca/nbody-relaxed"
BIN_DIR="$ROOT_DIR/bin"

################# write progenitors #########################
#./bin/remote.py \
#--cmd "$BIN_DIR/run_progenitors --write --cpus 10" \
#--job-dir "$BIN_DIR/temp" \
#--job-name "write-progenitors" \
#--nodes 1 --cpus 10 --ntasks 1 --cpus-per-task 1 --mem-per-cpu '6GB' --time '01:30'


################# summarize progenitors #########################
./bin/remote.py \
--cmd "$BIN_DIR/run_progenitors.py --summarize" \
--job-dir "$BIN_DIR/temp" \
--job-name "summarize-progs" \
--nodes 1 --cpus 1 --ntasks 1 --cpus-per-task 1 --mem-per-cpu '6GB' --time '01:00'

