import numpy as np
import matplotlib.pyplot as plt
from .results import RVResults, RandomProcessResults
from .sequences import InfiniteSequence

figure = plt.figure

xlabel = plt.xlabel
ylabel = plt.ylabel

xlim = plt.xlim
ylim = plt.ylim

def plot(*args, **kwargs):
    try:
        args[0].plot(**kwargs)
    except:
        plt.plot(*args, **kwargs)
        
        
