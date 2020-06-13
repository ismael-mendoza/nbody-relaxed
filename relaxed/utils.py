import csv
import gzip
from os.path import dirname
from pathlib import Path

root_path = Path(dirname(dirname(__file__)))
relaxed_path = root_path.joinpath("relaxed")
figure_path = root_path.joinpath("figures")
packages_path = root_path.joinpath("packages")
data_path = root_path.joinpath("data")
read_tree_path = packages_path.joinpath("consistent-trees/read_tree")


def is_iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


def get_stem(filename: str):
    return filename.split(".")[0]


def hlist_dat_to_csv(hlist_file):
    """
    Convert a hlist file to a .csv file.
    """
    hlist_file = Path(hlist_file)

    filename_stem = get_stem(hlist_file.name)
    new_filename = f"{filename_stem}.csv"
    hlist_new_file = hlist_file.parent.joinpath(new_filename)

    with gzip.open(hlist_file, "rt") as f:
        with open(hlist_new_file, mode="w") as csvfile:
            for i, line in enumerate(f):

                if i % 10000 == 0:  # show progress.
                    print(i)

                if i == 0:  # header
                    fieldnames = [
                        name[: name.rfind("(")].strip("#") for name in line.split()
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames)
                    writer.writeheader()

                if i >= 58:  # content.
                    dct = {key: value for key, value in zip(fieldnames, line.split())}
                    writer.writerow(dct)

                else:
                    continue  # skip descriptions.
