ROOT_DIR="/home/imendoza/alcca/nbody-relaxed"
BIN_DIR="$ROOT_DIR/bin"
OUTPUT="output_m11"
M_LOW=11.3
M_HIGH=11.6
N=10000
################# run pipeline #########################
./bin/catalog_pipeline --output-dir $OUTPUT make-ids --m-low $M_LOW --m-high $M_HIGH --n-haloes $N
./bin/catalog_pipeline --output-dir $OUTPUT make-dmcat
./bin/catalog_pipeline --output-dir $OUTPUT make-subhaloes
./bin/catalog_pipeline --output-dir $OUTPUT make-progenitors