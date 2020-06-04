# (1) Separate haloes in bins of 12, 13, 14 based on present day progenitor.

# (2) Obtain the redshift bins that all progenitors should share.
# in the paper they used 99 epochs:
# {0.06, 0.065, ..., 0.09, 0.095, 0.1, 0.11, 0.12, ..., 0.99, 1}.

# (3) Obtain mass fraction m = M(a)/M(0) as a function of scale (vice-versa)

# (4) Further subdivide each of the haloes in (1) into 20/30 logarithmic spaced bins based on
# log-mass. Should probably divide the whole table using something like cat[cat[Mvir] > m1 & cat[
# Mvir] < m2] for example. This will be a list with lists of catalogs in each of them.

# (5) For each of the bins (cats) in (4), extract the relevant property of interest and get the
# percentile rank of that property. You can added to the catalog along with the other properties.

# (6) Merge all the catalogs in that bin.

# (7) The above is the general procedure, you have to do this for all the progenitor cats first
# and then calculate the percentile rank for each scale slice.


