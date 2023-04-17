# Run the tests

import os
import numpy as np
from Util import Util
from CLT_class import CLT
from MIXTURE_CLT import MIXTURE_CLT

show_df = True

DATASET_DIR = "./dataset"
dataset_names = set(x.name[:x.name.index('.')]
                    for x in os.scandir(DATASET_DIR))

print(f"{len(dataset_names)} datasets: {dataset_names}")

# run each algorithm 5 times and compute the average and standard deviation

table = list()
for dataset_name in dataset_names:
    print(f"loading dataset [{dataset_name}]")
    test, train, valid = [
        Util.load_dataset(f"{DATASET_DIR}/{dataset_name}.{t}.data")
        for t in ("test", "ts", "valid")
    ]

    print("running CLT... ", end="")
    clt = CLT()
    clt.learn(train)
    ll = clt.computeLL(test) / test.shape[0]
    print(ll)
    table.append(("CLT", ll))

    print("running MIXTURE_CLT... ", end="")
    clt, ll = MIXTURE_CLT(), -np.inf
    for k in (2, 5, 10, 20):
        clt_ = MIXTURE_CLT()
        clt_.learn(train, n_components=k)
        # use validation dataset to compare K values
        ll_ = clt_.computeLL(valid) / valid.shape[0]
        # keep best performing model
        if ll_ > ll:
            clt, ll = clt_, ll_
    ll = clt.computeLL(test) / test.shape[0]
    print(ll)
    table.append(("MIXTURE_CLT", ll))

if show_df:
    import pandas
    print(pandas.DataFrame(table))