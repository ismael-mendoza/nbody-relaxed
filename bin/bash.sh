ROOT_DIR="/home/imendoza/alcca/nbody-relaxed"
NAME="m12"
OUTPUT="output_$NAME"
M_LOW=12.0
M_HIGH=12.2
N=10000
################# run pipeline #########################
#./bin/catalog_pipeline.py --output-dir $OUTPUT make-ids --m-low $M_LOW --m-high $M_HIGH \
#--n-haloes $N
#./bin/catalog_pipeline.py --output-dir $OUTPUT make-dmcat
#./bin/catalog_pipeline.py --output-dir $OUTPUT make-subhaloes
./bin/remote.py --cmd "./bin/catalog_pipeline.py --output-dir $OUTPUT make-progenitors" --jobname \
 "job_$NAME"




# M_LOW,M_HIGH = (11.3, 11.6), (12.0,12.2)