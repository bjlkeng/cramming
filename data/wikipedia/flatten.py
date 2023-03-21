""" 
    Simple script to convert JSON Wikipedia dump (from The Pile) to individual
    flat (non-json) files for easier processing
"""

import json

wikifiles = [
'/home/bjlkeng/devel/cramming/data/wikipedia/raw/wikipedia-en-0.json',
'/home/bjlkeng/devel/cramming/data/wikipedia/raw/wikipedia-en-1.json',
'/home/bjlkeng/devel/cramming/data/wikipedia/raw/wikipedia-en-2.json',
'/home/bjlkeng/devel/cramming/data/wikipedia/raw/wikipedia-en-3.json',
'/home/bjlkeng/devel/cramming/data/wikipedia/raw/wikipedia-en-4.json',
'/home/bjlkeng/devel/cramming/data/wikipedia/raw/wikipedia-en-5.json',
'/home/bjlkeng/devel/cramming/data/wikipedia/raw/wikipedia-en-6.json',
'/home/bjlkeng/devel/cramming/data/wikipedia/raw/wikipedia-en-7.json',
'/home/bjlkeng/devel/cramming/data/wikipedia/raw/wikipedia-en-8.json',
'/home/bjlkeng/devel/cramming/data/wikipedia/raw/wikipedia-en-9.json',
]

for i, wikifile in enumerate(wikifiles):
    with open(wikifile, 'r') as f:
      data = json.load(f)
    
    prefix = '/home/bjlkeng/devel/cramming/data/wikipedia/flat'
    batch = 200
    f = None
    for j, datum in enumerate(data):
        if j % batch == 0:
            if f is not None:
                f.close()
            outfile = f"{prefix}/wikifile-{i}-{j}.txt"
            f = open(outfile, 'w')
            print(f"Writing out {outfile} batch")

        f.write(datum + "[SEP]")
    f.close()