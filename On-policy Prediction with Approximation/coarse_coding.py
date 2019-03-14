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

def plot_subfigure(axes, row, col, y):
    axes[row,col].plot(y)

# returns 1 if float sample is within the tuple 'bounds'
def is_in(sample, bounds):
    return int(sample >= bounds[0] and sample < bounds[1])

# calculates left and right for approx_v for every value in samples
def get_all_bounds(width, offset, samples):
    bounds = {}
    for sample in samples:
        left, right = get_bounds(width, offset, sample)
        bounds[sample] = (left,right)
    return bounds
      
# calculates left and right for approx_v for a single sample
def get_bounds(width, offset, sample):
    right = int(sample/offset)  # rightmost field
    r_start = right*offset  # staring position of rightmost field
    remaining = width-(sample-r_start)  # remaining width to left
    remaining = min(r_start, remaining) # to handle left edge of axis
    left = right - int(remaining/offset)    # leftmost field
    if sample == left*offset + width: # handle right edge of field
        left += 1
    return (left,right)

# calculates v_hat=w*x
# bounds is pre-calculated left/right for known set of samples
def approx_v(w, sample, width, offset, bounds=None):
    if bounds is not None:
        left, right = bounds[sample]
    else:
        left, right = get_bounds(width, offset, sample)

    v = np.sum(w[left:right+1]) # w*x is sum of w's components
    return left,right,v

def plot_things():
    axes = setup_figure()
    for col,width in enumerate(widths):
        print(col)
        n = (200-width)//spacing+1  # number of features
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
    plt.show()

#plot_things()
