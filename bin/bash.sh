NAME="m12"
OUTPUT="output_$NAME"
JOBNAME="job_$NAME"
M_LOW=12.0
M_HIGH=12.2
N=10000
# M_LOW,M_HIGH = (11.15, 11.22), (12.0,12.2), (13, 14)

################# run pipeline #########################
./bin/catalog_pipeline.py --outdir $OUTPUT make-ids --m-low $M_LOW --m-high $M_HIGH \
--n-haloes $N
./bin/catalog_pipeline.py --outdir $OUTPUT make-dmcat
./bin/catalog_pipeline.py --outdir $OUTPUT make-subhaloes
./bin/remote.py --cmd "./bin/catalog_pipeline.py --outdir $OUTPUT make-progenitors && \
./bin/catalog_pipeline.py --outdir $OUTPUT combine-all" --jobname $JOBNAME