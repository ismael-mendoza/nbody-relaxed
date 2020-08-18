from ..catalogs import intersection


def add_progenitor_info(hcat, progenitor_file):
    # catalog with progenitor summary.
    cat = hcat.cat
    pcat = Table.read(progenitor_file)

    pcat = intersection(pcat, cat)
    cat = intersection(cat, pcat)

    fcat = astropy.table.join(cat, pcat, keys="id")
    hcat.cat = fcat
    return hcat
