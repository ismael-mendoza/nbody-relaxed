from os.path import dirname
from pathlib import Path

src_path = Path(dirname(dirname(__file__)))
root_path = Path(dirname(dirname(dirname(__file__))))
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
