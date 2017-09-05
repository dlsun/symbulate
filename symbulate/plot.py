import numpy as np
import matplotlib.pyplot as plt
from .sequences import InfiniteSequence

figure = plt.figure

xlabel = plt.xlabel
ylabel = plt.ylabel

xlim = plt.xlim
ylim = plt.ylim

def get_next_color(axes):
    color_cycle = axes._get_lines.prop_cycler
    color = next(color_cycle)["color"]
    return color

def configure_axes(axes, xdata, ydata, xlabel = None, ylabel = None):
    # Create 5% buffer on either end of plot so that leftmost and rightmost
    # lines are visible. However, if current axes are already bigger,
    # keep current axes.
    buff = .05 * (max(xdata) - min(xdata))
    xmin, xmax = axes.get_xlim()
    xmin = min(xmin, min(xdata) - buff)
    xmax = max(xmax, max(xdata) + buff)
    plt.xlim(xmin, xmax)

    _, ymax = axes.get_ylim()
    ymax = max(ymax, 1.05 * max(ydata))
    plt.ylim(0, ymax)
    
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

def plot(*args, **kwargs):
    try:
        args[0].plot(**kwargs)
    except:
        plt.plot(*args, **kwargs)
        
def discrete_check(heights):
    return sum([(i > 1) for i in heights]) > .8 * len(heights)
    
        
