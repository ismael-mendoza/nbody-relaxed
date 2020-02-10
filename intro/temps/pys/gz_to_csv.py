"""
Convert a hlist file to a .csv file.
"""

import gzip
import csv

base_file = '/home/imendoza/alcca/nbody-relaxed/intro/data/bolshoi/hlist_1.00035'

filename=f'{base_file}.list.gz'
new_filename = f'{base_file}.csv'

with gzip.open(filename,'rt') as f:
    with open(new_filename, mode='w') as csvfile: 
        for i,line in enumerate(f):

            #show progress. 
            if i%10000==0: 
                print(i)

            if i==0: #header 
                fieldnames = [name[:name.rfind('(')].strip('#') for name in line.split()]
                writer = csv.DictWriter(csvfile, fieldnames)
                writer.writeheader()

            if i>=58: #content. 
                dct = {key:value for key,value in zip(fieldnames, line.split())}
                writer.writerow(dct)

            else: 
                continue #skip descriptions.
