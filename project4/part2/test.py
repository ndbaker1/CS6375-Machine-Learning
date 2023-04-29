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

k_options = [2,5,10,20]
r_options = [0.05, 0.1, 0.2]
num_runs = 5

def load_dataset(dataset_name):
    print(f"loading dataset [{dataset_name}]")
    test, train, valid = [
        Util.load_dataset(f"{DATASET_DIR}/{dataset_name}.{t}.data")
        for t in ("test", "ts", "valid")
    ]
    return test, train, valid


for dataset_name in dataset_names:
    test, train, valid = load_dataset(dataset_name)
    print("running CLT... ", end="")
    clt = CLT()
    clt.learn(train)
    prev_ll = clt.computeLL(test) / test.shape[0]
    print("ll:", prev_ll)


for dataset_name in dataset_names:
    test, train, valid = load_dataset(dataset_name)
    print("running MIXTURE_CLT... ", end="")
    prev_ll, k_opt = -np.inf, 0
    for k in k_options:
        clt_ = MIXTURE_CLT()
        clt_.learn(train, n_components=k)
        # use validation dataset to compare K values
        ll_ = clt_.computeLL(valid) / valid.shape[0]
        # keep best performing model
        if ll_ > prev_ll:
            prev_ll, k_opt = ll_, k
    
    runs = []
    for _ in range(num_runs):
        clt = MIXTURE_CLT()
        clt.learn(test, n_components=k_opt)
        runs.append(clt.computeLL(test) / test.shape[0])
    mean_ll = np.mean(runs)
    std_ll = np.std(runs)
    print("ll:", mean_ll, "std:", std_ll, "k:", k_opt)


for dataset_name in dataset_names:
    test, train, valid = load_dataset(dataset_name)
    print("running RANDOM_FOREST_CLT... ", end="")
    prev_ll, k_opt, r_opt = -np.inf, 0, 0
    for k in k_options:
        for r in r_options:
            clt_ = RANDOM_FOREST_CLT()
            clt_.learn(train, k, r)
            # use validation dataset to compare K values
            ll_ = clt_.computeLL(valid) / valid.shape[0]
            # keep best performing model
            if ll_ > prev_ll:
                clt, prev_ll, k_opt, r_opt = clt_, ll_, k, r
    
    runs = []
    for _ in range(num_runs):
        clt = RANDOM_FOREST_CLT()
        clt.learn(test, k=k_opt, r=r_opt)
        runs.append(clt.computeLL(test) / test.shape[0])
    mean_ll = np.mean(runs)
    std_ll = np.std(runs)
    print("ll:", mean_ll, "std:", std_ll, "k:", k_opt, "r:", r_opt)
