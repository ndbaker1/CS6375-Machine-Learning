import pandas
import subprocess

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--run", action="store_true")
parser.add_argument("-i", "--iterations", type=int, default=1)
args = parser.parse_args()

iterations = args.iterations
run = args.run

if run:
    subprocess.call([
        "bash",
        "./compute_compression_ratio.sh",
        str(iterations),
    ])

df = pandas.DataFrame()

with open("./Koala.csv") as f:
    idf = pandas.read_csv(f)
    idf = idf.assign(label="koala")
    df = pandas.concat([df, idf])

with open("./Penguins.csv") as f:
    idf = pandas.read_csv(f)
    idf = idf.assign(label="penguins")
    df = pandas.concat([df, idf])

print(df.groupby(["label", "k"]).mean().round(2))
