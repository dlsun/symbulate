import numpy as np
import matplotlib.pyplot as plt
from .sequences import InfiniteSequence
from scipy.stats import gaussian_kde

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
        
def is_discrete(heights):
    return sum([(i > 1) for i in heights]) > .8 * len(heights)

def count_var(x):
    counts = {}
    for val in x:
        if val in counts:
            counts[val] += 1
        else:
            counts[val] = 1
    return counts
    
def compute_density(values):
	density = gaussian_kde(values)
	density.covariance_factor = lambda: 0.25
	density._compute_covariance()
	return density
    
def add_colorbar(fig, type, mappable):
    if 'marginal' not in type:
        caxes = fig.add_axes([0, 0.1, 0.05, 0.8])
    else:
        caxes = fig.add_axes([0, 0.1, 0.05, 0.57])
        
    cbar = plt.colorbar(mappable=mappable, cax=caxes)
    caxes.yaxis.set_ticks_position('left')
    cbar.set_label('Density')
    caxes.yaxis.set_label_position('left')
    return cbar, caxes

def setup_tile(v, bins, discrete):
    if not discrete:
        v_bin = np.linspace(min(v), max(v) + 1, bins)
        v_lab = [min(v)]
        for i in range(len(v_bin) - 1):
            v_lab.append(0.5 * (v_bin[i] + v_bin[i+1]))
        v_pos = range(len(v_lab))
        v_vect = np.digitize(v, v_bin)
    else:
        v_lab = np.unique(v)
        v_pos = range(len(v_lab))
        v_map = dict(zip(v_lab, v_pos))
        v_vect = np.vectorize(v_map.get)(v)
    return v_vect, v_lab, v_pos

def make_tile(x, y, bins, discrete_x, discrete_y, ax):
    x_vect, x_lab, x_pos = setup_tile(x, bins, discrete_x)
    y_vect, y_lab, y_pos = setup_tile(y, bins, discrete_y)
    nums = len(x_vect)
    counts = count_var(list(zip(y_vect, x_vect)))
    intensity = np.zeros(shape=(len(y_lab), len(x_lab)))
        
    for key, val in counts.items():
        intensity[key] = val / nums
    hm = ax.matshow(intensity, cmap='Blues', origin='lower', aspect='auto')
    ax.xaxis.set_ticks_position('bottom')
    if not discrete_x: x_lab = np.around(x_lab, decimals=1)
    if not discrete_y: y_lab = np.around(y_lab, decimals=1)
    return hm, x_lab, y_lab, x_pos, y_pos

def setup_ticks(pos, lab, ax, axis):
    if axis == 'x':
        ax.set_xticks(pos)
        ax.set_xticklabels(lab)
    elif axis == 'y':
        ax.set_yticks(pos)
        ax.set_yticklabels(lab)

def make_violin(data, positions, ax, axis, alpha):
    values = []
    if axis == 'x':
        for i in positions:
            values.append(data[data[:, 0] == i, 1].tolist())
        violins = ax.violinplot(dataset=values, showmedians=True)
        ax.set_xticks(np.array(positions) + 1)
        ax.set_xticklabels(positions)
    elif axis == 'y':
        for i in positions:
            values.append(data[data[:, 1] == i, 0].tolist())
        violins = ax.violinplot(dataset=values, showmedians=True, vert=False)
        ax.set_yticks(np.array(positions) + 1)
        ax.set_yticklabels(positions)
    for part in violins['bodies']:
        part.set_edgecolor('black')
        part.set_alpha(alpha)
    for part in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = violins[part]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
    return violins

def marginal_impulse(count, height, color, ax_marg, alpha, axis):
    key = list(count.keys())
    val = list(height)
    tot = sum(val)
    val = [i / tot for i in val]
    if axis == 'x':
        ax_marg.vlines(key, 0, val, color=color, alpha=alpha)
    elif axis == 'y':
        ax_marg.hlines(key, 0, val, color=color, alpha=alpha)

def density2D(x, y, ax):
    res = np.vstack([x, y])
    density = gaussian_kde(res)
    xmax, xmin = max(x), min(x)
    ymax, ymin = max(y), min(y)
    Xgrid, Ygrid = np.meshgrid(np.linspace(xmin, xmax, 100),
                               np.linspace(ymin, ymax, 100))
    Z = density.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    den = ax.imshow(Z.reshape(Xgrid.shape), origin='lower', cmap='Blues',
              aspect='auto', extent=[xmin, xmax, ymin, ymax]
    )
    return den

