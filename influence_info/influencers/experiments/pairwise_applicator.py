import numpy as np
import os, json

def pairwise_applicator(func, agg) :
    def applicator(x, y) :
        values = [func(x_i, y_i) for x_i, y_i in zip(x, y)]
        return agg(values)

    return applicator

def mean_pairwise_applicator(func) :
    return pairwise_applicator(func, np.mean)

def load_influence_values(folder) :
    if_values = np.load(os.path.join(folder, "influence_values.npy"))
    training_idx = np.argsort(json.load(open(os.path.join(folder, "training_idx.json"))))
    validation_idx = np.argsort(json.load(open(os.path.join(folder, "validation_idx.json"))))

    if_values = if_values[validation_idx]
    if_values = if_values[:, training_idx]

    return if_values

def parse(output_folder, method) :
    relpath = os.path.relpath(method, output_folder)
    if "/" in relpath :
        methodname, args = relpath.split("/")
    else :
        methodname, args = relpath, ""
    return methodname, args

import glob

def get_all_values(output_folder, regexes) :
    all_methods = [path for regex in regexes for path in glob.glob(os.path.join(output_folder, regex))]
    print(all_methods)
    if_values = []
    for method in all_methods :
        name, args = parse(output_folder, method)
        if_values.append((name, args, load_influence_values(method)))

    return if_values

def pairwise_experiment(if_values, applicator) :
    num_methods = len(if_values) 

    triplets = []
    for i in range(num_methods) :
        for j in range(num_methods) :
            value = applicator(if_values[i][2], if_values[j][2])
            triplets.append((if_values[i][:2], if_values[j][:2], value))

    return triplets

collapse = lambda name, args : f"{name}/{args}"

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairwise_matrix(triplets) :
    x, y, z = zip(*triplets)
    x = [collapse(*k) for k in x]
    y = [collapse(*k) for k in y]
    data = pd.DataFrame(data={"m1" : x, "m2": y, "value": z})
    data = data.pivot(index="m1", columns="m2", values="value")
    sns.heatmap(data, cmap="Blues")
    plt.show()





