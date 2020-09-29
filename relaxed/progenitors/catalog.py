from ..halo_catalogs import intersection
import astropy.table
from astropy.table import Table


def add_progenitor_info(hcat, summary_file):
    # summary file: catalog with progenitor summary.
    cat = hcat.cat
    pcat = Table.read(summary_file)
    assert set(pcat.colnames).intersection(set(cat.colnames)).pop() == "id"

    pcat = intersection(pcat, cat)
    cat = intersection(cat, pcat)

    fcat = astropy.table.join(cat, pcat, keys="id")
    hcat.cat = fcat
    return hcat
