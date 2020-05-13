import re
from astropy.table import Table


def get_prog_lines_generator(progenitor_file):
    # read file line by line
    prog_line = None

    with open(progenitor_file, 'r') as pf:
        for line in pf:
            line = line.rstrip()  # remove trailing whitespace
            if line:  # not empty
                tree_root_match = re.match(r"# tree root id: (\d+) #", line)
                halo_match = re.match(r"(\d+),(\d+),(\d+),(\d*),(\d*),(\d*)", line)
                weird_match = re.match(r"id=\d+, mmp=(\d+)", line)
                total_halos_match = re.match(r"Number of root nodes is: (\d+)", line)
                total_root_halos_match = re.match(r"final count is: (\d+)", line)

                if tree_root_match:
                    root_id = tree_root_match.groups()[0]
                    prog_line = ProgenitorLine(root_id=root_id)

                elif halo_match:
                    halo_id, mvir, scale, coprog_id, coprog_mvir, coprog_scale = (float(x) if x != '' else -1 for x in
                                                                                  halo_match.groups())
                    prog_line.add((halo_id, mvir, scale, coprog_id, coprog_mvir, coprog_scale))

                elif weird_match:
                    assert weird_match.groups()[1] == '0', "Expected failure should have this format"

                elif total_halos_match or total_root_halos_match:
                    # we are done
                    break

                else:
                    raise ValueError("Something is wrong in this halo file.")

            elif prog_line is not None:
                prog_line.finalize()
                yield prog_line
                prog_line = None

            else:
                continue


class ProgenitorLine(object):
    def __init__(self, root_id):
        """
        Class representing progenitors read from the file created by save_progenitors.py
        """
        self.root_id = root_id
        self.cat = Table()
        self.colnames = ['halo_id', 'mvir', 'scale', 'coprog_ids', 'coprog_mvirs', 'coprog_scale']
        self.rows = []

    def add(self, halo_tuple):
        self.rows.append(halo_tuple)

    def finalize(self):
        self.cat = Table(rows=self.rows, names=self.colnames)
