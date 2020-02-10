import argparse
import ytree
import numpy as np
from pathlib import Path
import csv


def find_zform_dx(mvirs):
    return np.argmax(mvirs < mvirs[0]/2)


parser = argparse.ArgumentParser(description='Obtain the main progenitor line from a single tree.')
parser.add_argument('--tree-file', type=str, required=True)

args = parser.parse_args()
tree_file = Path(args.tree_file)
progenitor_path = tree_file.parent.joinpath("progenitors")

a = ytree.load(tree_file.as_posix())
a.set_selector("max_field_value", "mvir")  # change selector b/c ytree uses 'mass' by default.
results_file_name = f"arr_{tree_file.name}.csv".replace(".dat", "")
results_file = progenitor_path.joinpath(results_file_name)
log_file = "/home/imendoza/alcca/nbody-relaxed/intro/log.txt"

# arrs = []

with open(results_file.as_posix(), mode='w') as csvfile:
    fieldnames = ['root_id', 'mvir2', 'zform']  # mvir2 refers to the second largest virial mass.
    writer = csv.DictWriter(csvfile, fieldnames)
    writer.writeheader()
    csvfile.flush()
    print(results_file.as_posix(), file=open(log_file,'a'))

    for i, tree in enumerate(a):
        if i % 1000 == 0:
            print(f"{i}", file=open(log_file,'a'))

        root_id = int(tree['id'])
        mvirs = np.array(tree['prog', 'mvir'], dtype=np.float32)  # extract progenitor line mass and scale.
        scales = np.array(tree['prog', 'scale'], dtype=np.float32)
        zform_idx = find_zform_dx(mvirs)
        dct = {
            'root_id': root_id,
            'mvir2': mvirs[zform_idx],
            'zform': scales[zform_idx]
        }
        writer.writerow(dct)
        csvfile.flush()  # make sure we are not storing info but actually write to file.

        # arr = []
        #
        # arr.append(root_id)
        # arr.append(mvirs[zform_idx])
        # arr.append(scales[zform_idx])
        #
        # arrs.append(np.array(arr))

# arrs = np.array(arrs)
# np.save(arr_file.as_posix(), arrs)
