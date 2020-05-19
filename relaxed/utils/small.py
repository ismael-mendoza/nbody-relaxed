import csv
import gzip
from pathlib import Path


def get_stem(filename: str):
    return filename.split('.')[0]


def hlist_dat_to_csv(hlist_file):
    """
    Convert a hlist file to a .csv file.
    """
    hlist_file = Path(hlist_file)

    filename_stem = get_stem(hlist_file.name)
    new_filename = f'{filename_stem}.csv'
    hlist_new_file = hlist_file.parent.joinpath(new_filename)

    with gzip.open(hlist_file, 'rt') as f:
        with open(hlist_new_file, mode='w') as csvfile:
            for i, line in enumerate(f):

                if i % 10000 == 0:  # show progress.
                    print(i)

                if i == 0:  # header
                    fieldnames = [name[:name.rfind('(')].strip('#') for name in
                                  line.split()]
                    writer = csv.DictWriter(csvfile, fieldnames)
                    writer.writeheader()

                if i >= 58:  # content.
                    dct = {key: value for key, value in zip(fieldnames, line.split())}
                    writer.writerow(dct)

                else:
                    continue  # skip descriptions.
