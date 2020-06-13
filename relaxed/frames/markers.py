# (1) Separate haloes in bins of 12, 13, 14 based on present day progenitor.
# Q: why narrow bin for the first two catalogs.
# making a narrow bin will drastically reduce the number of tables.
# do this initial selection ~50k (something manageable) put it to a pandas data frame.

# (2) Obtain the redshift bins that all progenitors should share.
# in the paper they used 99 epochs:
# keep all the data , and add NaNs for the ones that don't have that.
# {0.06, 0.065, ..., 0.09, 0.095, 0.1, 0.11, 0.12, ..., 0.99, 1}.

# (3) Obtain mass fraction m = M(a)/M(0) as a function of scale (vice-versa)

# ===> get spearman correlation m(a) without the markers.

# (4) Further subdivide each of the haloes in (1) into 20/30 logarithmic spaced bins (will
# have to decide what bins to use) based on
# log-mass. Should probably divide the whole table using something like
# pd[pd[Mvir] > m1 & pd[Mvir] < m2]
# for example. This will be a list with lists of catalogs in each of them.


# (5) For each of the bins (cats) in (4), extract the relevant property of interest and get the
# percentile rank of that property. You can added to the catalog along with the other properties.

# (7) The above is the general procedure, you have to do this for all the progenitor cats first
# and then calculate the percentile rank for each scale slice.

# pandas table
# * a column entry can be pandas.


# TODO: Write each of the m(a) for each a (99 ?) to a separate file
