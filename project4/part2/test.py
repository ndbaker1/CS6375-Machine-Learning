# Run the tests

import os
import numpy as np
from Util import Util
from CLT_class import CLT
from MIXTURE_CLT import MIXTURE_CLT
from RANDOM_FOREST_CLT import RANDOM_FOREST_CLT

show_df = True

DATASET_DIR = "./dataset"
dataset_names = set(x.name[:x.name.index('.')]
                    for x in os.scandir(DATASET_DIR))

print(f"{len(dataset_names)} datasets: {dataset_names}")

# run each algorithm 5 times and compute the average and standard deviation

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
    print("ll:", ll)

    print("running MIXTURE_CLT... ", end="")
    clt, ll, k_opt = MIXTURE_CLT(), -np.inf, 0
    for k in (2, 5, 10, 20):
        clt_ = MIXTURE_CLT()
        clt_.learn(train, n_components=k)
        # use validation dataset to compare K values
        ll_ = clt_.computeLL(valid) / valid.shape[0]
        # keep best performing model
        if ll_ > ll:
            clt, ll, k_opt = clt_, ll_, k
    ll = clt.computeLL(test) / test.shape[0]
    print("ll:", ll, "k:", k_opt)

    print("running RANDOM_FOREST_CLT... ", end="")
    clt, ll, k_opt, r_opt = RANDOM_FOREST_CLT(), -np.inf, 0, 0
    for k in (2, 5, 10, 20):
        for r in (0.05, 0.1, 0.2):
            clt_ = RANDOM_FOREST_CLT()
            clt_.learn(train, k, r)
            # use validation dataset to compare K values
            ll_ = clt_.computeLL(valid) / valid.shape[0]
            # keep best performing model
            if ll_ > ll:
                clt, ll, k_opt, r_opt = clt_, ll_, k, r
    ll = clt.computeLL(test) / test.shape[0]
    print("ll:", ll, "k:", k_opt, "r:", r_opt)
