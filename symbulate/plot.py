import numpy as np
import matplotlib.pyplot as plt
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

def setup_ticks(pos, lab, ax):
    ax.set_ticks(pos)
    ax.set_ticklabels(lab)
    
def add_colorbar(fig, type, mappable, label):
    #create axis for cbar to place on left
    if 'marginal' not in type: 
        caxes = fig.add_axes([0, 0.1, 0.05, 0.8])
    else: #adjust height if marginals
        caxes = fig.add_axes([0, 0.1, 0.05, 0.57])
    cbar = plt.colorbar(mappable=mappable, cax=caxes)
    caxes.yaxis.set_ticks_position('left')
    cbar.set_label(label)
    caxes.yaxis.set_label_position('left')
    return caxes

def setup_tile(v, bins, discrete):
    if not discrete:
        v_lab = np.linspace(min(v), max(v), bins + 1)
        v_pos = np.arange(0, len(v_lab)) - 0.5
        v_vect = np.digitize(v, v_lab, right=True) - 1
    else:
        v_lab = np.unique(v) #returns sorted array
        v_pos = range(len(v_lab))
        v_map = dict(zip(v_lab, v_pos))
        v_vect = np.vectorize(v_map.get)(v)
    return v_vect, v_lab, v_pos

def make_tile(x, y, bins, discrete_x, discrete_y, ax):
    x_vect, x_lab, x_pos = setup_tile(x, bins, discrete_x)
    y_vect, y_lab, y_pos = setup_tile(y, bins, discrete_y)
    nums = len(x_vect)
    counts = count_var(list(zip(y_vect, x_vect)))
    y_shape = len(y_lab) if discrete_y else len(y_lab) - 1
    x_shape = len(x_lab) if discrete_x else len(x_lab) - 1
    intensity = np.zeros(shape=(y_shape, x_shape))
        
    for key, val in counts.items():
        intensity[key] = val / nums
    if not discrete_x: x_lab = np.around(x_lab, decimals=1)
    if not discrete_y: y_lab = np.around(y_lab, decimals=1)
    hm = ax.matshow(intensity, cmap='Blues', origin='lower', aspect='auto', vmin=0)
    ax.xaxis.set_ticks_position('bottom')
    setup_ticks(x_pos, x_lab, ax.xaxis)
    setup_ticks(y_pos, y_lab, ax.yaxis)
    return hm

def make_violin(data, positions, ax, axis, alpha):
    values = []
    i, j = (0, 1) if axis == 'x' else (1, 0)
    values = [data[data[:, i] == pos, j].tolist() for pos in positions]
    violins = ax.violinplot(dataset=values, showmedians=True,
                            vert=False if axis == 'y' else True)
    setup_ticks(np.array(positions) + 1, positions, 
                ax.xaxis if axis == 'x' else ax.yaxis)
    for part in violins['bodies']:
        part.set_edgecolor('black')
        part.set_alpha(alpha)
    for component in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = violins[component]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)

def make_marginal_impulse(count, color, ax_marg, alpha, axis):
    key, val = list(count.keys()), list(count.values())
    tot = sum(val)
    val = [i / tot for i in val]
    if axis == 'x':
        ax_marg.vlines(key, 0, val, color=color, alpha=alpha)
    elif axis == 'y':
        ax_marg.hlines(key, 0, val, color=color, alpha=alpha)

def make_density2D(x, y, ax):
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
