from pathlib import Path
from os.path import dirname

src_path = Path(dirname(__file__))
root_path = Path(dirname(dirname(__file__)))
figure_path = root_path.joinpath("figures")
packages_path = root_path.joinpath("packages")
data_path = root_path.joinpath("data")
