# to recreate Figure 9.8
# This is a newer version of this algorithm.
# I suspect that there's something wrong with the boundary conditions

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

def plot_subfigure(axes, row, col, y):
    axes[row,col].plot(y)

# returns 1 if float sample is within the tuple 'bounds'
def is_in(sample, bounds):
    return int(sample >= bounds[0] and sample < bounds[1])

# calculates left and right for approx_v for a single sample
def get_bounds(width, offset, sample):
    right = int(sample/offset)  # rightmost field
    left = right - (width//offset - 1)
    left = max(left, 0)
    if sample == left*offset + width: # handle right edge of field
        left += 1
    return (left,right)

# calculates v_hat=w*x
# bounds is pre-calculated left/right for known set of samples
def approx_v(w, sample, width, offset):
    left, right = get_bounds(width, offset, sample)
    v = np.sum(w[left:right+1]) # w*x is sum of w's components
    return left,right,v

def plot_things():
    axes = setup_figure()
    for col,width in enumerate(widths):
        print(col)
        n = 200//spacing  # number of features
        alpha = 1/n
        w = np.zeros(n) # weights to learn
        row = 0
        for step in range(examples[-1]):
            if step+1 in examples:
                x = np.linspace(0,200,1000)
                y = [approx_v(w,i,width,spacing)[2] for i in x]
                plot_subfigure(axes, row, col, y)
                row += 1
            sample = env.np_random.random_sample()*200
            true_value = is_in(sample, (50, 150))   # value of square function
            left,right,v = approx_v(w,sample,width,spacing)
            w[left:right+1] += alpha*(true_value - v)
    print(w)
    plt.show()

plot_things()
