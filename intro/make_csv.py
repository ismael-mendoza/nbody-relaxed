import gzip
import csv
import os

#base = 'data/hlist_1.00109'
base = 'data/hlist_1.00109'

filename=f'{base}.list.gz'
new_filename = f'{base}.csv'

# def create_csv(csv_filename, dcts, fieldnames):
#     with open(csv_filename, mode='w') as csv_file:
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

#         writer.writeheader()
#         for dct in dcts:
#             writer.writerow(dct)

# #add new rows to existing csv file. 
# def append_csv(csv_filename, dcts, fieldnames):
#     assert os.path.isfile(csv_filename), "CSV file does not exist."

#     with open(csv_filename, 'a', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames)

#         for dct in dcts:
#             writer.writerow(dct)

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
