#plan 
# (1) read line by line 
# (2) paste into a CSV format, 
# (3) use: https://docs.astropy.org/en/stable/io/ascii/read.html#reading-large-tables-in-chunks; to read it 
import gzip

with gzip.open(filename,'rt') as f:
    arr = []
    limit=int(1e7)
    for i,line in enumerate(f):
        arr.append(line)
        if i==limit:
           break
