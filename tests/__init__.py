from os.path import dirname
from pathlib import Path
import sys

project_path = Path(dirname(dirname(__file__)))  # path to nbody-relaxed
sys.path.insert(0, project_path)
