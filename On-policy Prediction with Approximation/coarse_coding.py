# to recreate Figure 9.8

import numpy as np
import matplotlib.pyplot as plt
from random_walk_env import RandomWalkEnv
env = RandomWalkEnv()

# tile dimensions
widths = [20, 45, 100]
spacing = 2
examples = [10, 40, 160, 640, 2560, 10240]


def setup_figure():
    fig, axes = plt.subplots(6, 3, True, True)
    for ax in axes.flatten():
        ax.axis('off')  # removes everything but plotted data
    return axes

def plot_subfigure(row, col, y):
    axes[row,col].plot(y)

# returns 1 if float sample is within the tuple 'bounds'
def is_in(sample, bounds):
    return int(sample >= bounds[0] and sample < bounds[1])

# I'm sure there's a more efficient way to do this, 
#   but efficiency is not that important to me right now
def get_features(sample, receptive_fields):
    features = [is_in(sample, field) for field in receptive_fields]
    return np.array(features)

def get_receptive_fields(width, num_features, offset):
    return [(i*offset,i*offset+width) for i in range(num_features)]

def plot_things():
    axes = setup_figure()
    for col,width in enumerate(widths):
        print(col)
        n = (200-width)//spacing+1  # number of features
        receptive_fields = get_receptive_fields(width, n, spacing)
        alpha = 1/n
        w = np.zeros(n) # weights to learn
        row = 0
        for step in range(examples[-1]):
            if step+1 in examples:
                x = np.linspace(0,200,1000)
                y = [np.dot(w, get_features(i,receptive_fields)) for i in x]
                plot_subfigure(row, col, y)
                row += 1
            sample = env.np_random.random_sample()*200
            true_value = is_in(sample, (50, 150))   # value of square function
            features = get_features(sample, receptive_fields)
            w += alpha*(true_value - np.dot(w, features))*features
    plt.show()
