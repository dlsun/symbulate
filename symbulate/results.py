"""Data structures for storing the results of a simulation.

This module provides data structures for storing the
results of a simulation, either outcomes from a
probability space or realizations of a random variable /
random process.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from numbers import Number
from collections import Counter

from .sequences import TimeFunction
from .table import Table
from .utils import is_scalar, is_vector, get_dimension
from .plot import configure_axes, get_next_color, discrete_check
from statsmodels.graphics.mosaicplot import mosaic
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import Affine2D
from scipy.stats import gaussian_kde

plt.style.use('seaborn-colorblind')

def is_hashable(x):
    return x.__hash__ is not None

def count_var(x):
    counts = {}
    for val in x:
        if val in counts:
            counts[val] += 1
        else:
            counts[val] = 1
    return counts

class Results(list):

    def __init__(self, results):
        for result in results:
            self.append(result)

    def apply(self, fun):
        """Apply a function to each outcome of a simulation.

        Args:
          fun: A function to apply to each outcome.

        Returns:
          Results: A Results object of the same length,
            where each outcome is the result of applying
            the function to each outcome from the original
            Results object.
        """
        return type(self)(fun(x) for x in self)

    def __getitem__(self, i):
        return self.apply(lambda x: x[i])

    def get(self, i):
        for j, x in enumerate(self):
            if j == i:
                return x

    def _get_counts(self):
        counts = {}
        for x in self:
            if is_hashable(x):
                y = x
            elif isinstance(x, list) and all(is_hashable(i) for i in x):
                y = tuple(x)
            else:
                y = str(x)
            if y in counts:
                counts[y] += 1
            else:
                counts[y] = 1
        return counts

    def tabulate(self, outcomes=None, normalize=False):
        """Counts up how much of each outcome there were.

        Args:
          outcomes (list): A list of outcomes to tabulate.
            By default, will tabulate all outcomes that
            appear in the Results.  Use this option if
            you want to include outcomes that could
            potentially not appear in the Results.
          normalize (bool): If True, return the relative
            frequency. Otherwise, return the counts.
            Defaults to False.

        Returns:
          Table: A Table with each of the observed
            outcomes and their freuencies.
        """
        table = Table(self._get_counts(), outcomes)
        if normalize:
            return table / len(self)
        else:
            return table


    # The following functions return a Results object
    # with the outcomes that satisfy a given criterion.

    def filter(self, fun):
        """Filters the results of a simulation and
             returns only those outcomes that satisfy
             a given criterion.

        Args:
          fun (outcome -> bool): A function that
            takes in an outcome and returns a
            True / False. Only the outcomes that
            return True will be kept; the others
            will be filtered out.

        Returns:
          Results: Another Results object containing
            only those outcomes for which the function
            returned True.
        """
        return type(self)(x for x in self if fun(x))

    def filter_eq(self, value):
        return self.filter(lambda x: x == value)

    def filter_neq(self, value):
        return self.filter(lambda x: x != value)

    def filter_lt(self, value):
        return self.filter(lambda x: x < value)

    def filter_leq(self, value):
        return self.filter(lambda x: x <= value)

    def filter_gt(self, value):
        return self.filter(lambda x: x > value)

    def filter_geq(self, value):
        return self.filter(lambda x: x >= value)


    # The following functions return an integer indicating
    # how many outcomes passed a given criterion.

    def count(self, fun=lambda x: True):
        """Counts the number of outcomes that satisfied
             a given criterion.

        Args:
          fun (outcome -> bool): A function that
            takes in an outcome and returns a
            True / False. Only the outcomes that
            return True will be counted.

        Returns:
          int: The number of outcomes for which
            the function returned True.
        """
        return len(self.filter(fun))

    def count_eq(self, value):
        return len(self.filter_eq(value))

    def count_neq(self, value):
        return len(self.filter_neq(value))

    def count_lt(self, value):
        return len(self.filter_lt(value))

    def count_leq(self, value):
        return len(self.filter_leq(value))

    def count_gt(self, value):
        return len(self.filter_gt(value))

    def count_geq(self, value):
        return len(self.filter_geq(value))


    # The following functions define vectorized operations
    # on the Results object.

    def __eq__(self, other):
        return self.apply(lambda x: x == other)

    def __ne__(self, other):
        return self.apply(lambda x: x != other)

    def __lt__(self, other):
        return self.apply(lambda x: x < other)

    def __le__(self, other):
        return self.apply(lambda x: x <= other)

    def __gt__(self, other):
        return self.apply(lambda x: x > other)

    def __ge__(self, other):
        return self.apply(lambda x: x >= other)


    def plot(self):
        raise Exception("Only simulations of random variables (RV) "
                        "can be plotted, but you simulated from a " 
                        "probability space. You must first define a RV "
                        "on your probability space and simulate it. "
                        "Then call .plot() on those simulations.")
 
    def mean(self):
        raise Exception("You can only call .mean() on simulations of "
                        "random variables (RV), but you simulated from "
                        "a probability space. You must first define "
                        "a RV on your probability space and simulate it "
                        "Then call .mean() on those simulations.")

    def var(self):
        raise Exception("You can only call .var() on simulations of "
                        "random variables (RV), but you simulated from "
                        "a probability space. You must first define "
                        " a RV on your probability space and simulate it "
                        "Then call .var() on those simulations.")

    def sd(self):
        raise Exception("You can only call .sd() on simulations of "
                        "random variables (RV), but you simulated from "
                        "a probability space. You must first define "
                        " a RV on your probability space and simulate it "
                        "Then call .sd() on those simulations.")

    def corr(self):
        raise Exception("You can only call .corr() on simulations of "
                        "random variables (RV), but you simulated from "
                        "a probability space. You must first define "
                        " a RV on your probability space and simulate it "
                        "Then call .corr() on those simulations.")
   
    def cov(self):
        raise Exception("You can only call .cov() on simulations of "
                        "random variables (RV), but you simulated from "
                        "a probability space. You must first define "
                        " a RV on your probability space and simulate it "
                        "Then call .cov() on those simulations.")


    def _repr_html_(self):

        table_template = '''
    <table>
      <thead>
        <th width="10%">Index</th>
        <th width="90%">Result</th>
      </thead>
      <tbody>
        {table_body}
      </tbody>
    </table>
    '''
        row_template = '''
        <tr>
          <td>%s</td><td>%s</td>
        </tr>
        '''

        def truncate(result):
            if len(result) > 100:
                return result[:100] + "..."
            else:
                return result

        table_body = ""
        for i, x in enumerate(self):
            table_body += row_template % (i, truncate(str(x)))
            # if we've already printed 9 rows, skip to end
            if i >= 8:
                table_body += "<tr><td>...</td><td>...</td></tr>"
                i_last = len(self) - 1
                table_body += row_template % (i_last, truncate(str(self.get(i_last))))
                break
        return table_template.format(table_body=table_body)


class RVResults(Results):

    def plot(self, type=None, alpha=None, normalize=True, jitter=False, 
        bins=None, **kwargs):
        if type is not None:
            if isinstance(type, str):
                type = (type,)
            elif not isinstance(type, (tuple, list)):
                raise Exception("I don't know how to plot a " + str(type))
        
        dim = get_dimension(self)
        if dim == 1:
            counts = self._get_counts()
            heights = counts.values()
            discrete = discrete_check(heights)
            if type is None:
                if discrete:
                    type = ("impulse",)
                else:
                    type = ("hist",)
            if alpha is None:
                alpha = .5
            if bins is None:
                bins = 30

            fig = plt.gcf()
            ax = plt.gca()
            color = get_next_color(ax)
            
            if 'density' in type:
                if discrete:
                    xs = sorted(list(counts.keys()))
                    ys = []
                    for val in xs:
                        ys.append(counts[val] / len(self))
                    ax.plot(xs, ys, marker='o', color=color, linestyle='-')
                    if len(type) == 1:
                        plt.ylabel('Relative Frequency')
                else:
                    density = gaussian_kde(self)
                    density.covariance_factor = lambda: 0.25
                    density._compute_covariance()
                    xs = np.linspace(min(self), max(self), 1000)
                    ax.plot(xs, density(xs), linewidth=2, color=color)
                    if len(type) == 1 or (len(type) == 2 and 'rug' in type):
                        plt.ylabel('Density')

            if 'hist' in type or 'bar' in type:
                ax.hist(self, color=color, bins=bins, alpha=alpha, normed=True, **kwargs)
                plt.ylabel("Density" if normalize else "Count")
            elif 'impulse' in type:
                x = list(counts.keys())
                y = list(counts.values())
                if alpha is None:
                    alpha = .7
                if normalize:
                    y_tot = sum(y)
                    y = [i / y_tot for i in y]
                if jitter:
                    a = .02 * (max(x) - min(x))
                    noise = np.random.uniform(low=-a, high=a)
                    x = [i + noise for i in x]
                # plot the impulses
                ax.vlines(x, 0, y, color=color, alpha=alpha, **kwargs)
                configure_axes(ax, x, y, ylabel="Relative Frequency" if normalize else "Count")
            if 'rug' in type:
                if discrete:
                    self = self + np.random.normal(loc=0, scale=.002 * (max(self) - min(self)), size=len(self))
                ax.plot(list(self), [0.001]*len(self), '|', linewidth = 5, color='k')
                if len(type) == 1:
                    ax.yaxis.set_ticklabels([])
                    ax.yaxis.set_ticks([])
        elif dim == 2:
            x, y = zip(*self)

            x_count = count_var(x)
            y_count = count_var(y)
            x_height = x_count.values()
            y_height = y_count.values()
            
            discrete_x = discrete_check(x_height)
            discrete_y = discrete_check(y_height)

            if type is None:
                type = ("scatter",)

            if alpha is None:
                alpha = .5

            if bins is None:
                if 'mixed-tile' in type:
                    bins = 10
                else:
                    bins = 30

            if 'tile' in type and (not discrete_x or not discrete_y):
                print('type=\'tile\' is only valid for 2 discrete variables')
                type = ['scatter' if x == 'tile' else x for x in list(type)]

            if 'marginal' in type:
                fig = plt.gcf()
                gs = GridSpec(4, 4)
                ax = fig.add_subplot(gs[1:4, 0:3])
                ax_marg_x = fig.add_subplot(gs[0, 0:3])
                ax_marg_y = fig.add_subplot(gs[1:4, 3])
                color = get_next_color(ax)
                if 'density' in type:
                    densityX = gaussian_kde(x)
                    densityX.covariance_factor = lambda: 0.25
                    densityX._compute_covariance()
                    densityY = gaussian_kde(y)
                    densityY.covariance_factor = lambda: 0.25
                    densityY._compute_covariance()
                    x_lines = np.linspace(min(x), max(x), 1000)
                    y_lines = np.linspace(min(y), max(y), 1000)
                    ax_marg_x.plot(x_lines, densityX(x_lines), linewidth=2, color=get_next_color(ax))
                    ax_marg_y.plot(y_lines, densityY(y_lines), linewidth=2, color=get_next_color(ax), 
                                  transform=Affine2D().rotate_deg(270) + ax_marg_y.transData)
                else:
                    if discrete_x:
                        x_key = list(x_count.keys())
                        x_val = list(x_height)
                        x_tot = sum(x_val)
                        x_val = [i / x_tot for i in x_val]
                        ax_marg_x.vlines(x_key, 0, x_val, color=get_next_color(ax), alpha=alpha, **kwargs)
                    else:
                        ax_marg_x.hist(x, color=get_next_color(ax), normed=True, 
                                       alpha=alpha, bins=bins)
                    if discrete_y:
                        y_key = list(y_count.keys())
                        y_val = list(y_height)
                        y_tot = sum(y_val)
                        y_val = [i / y_tot for i in y_val]
                        ax_marg_y.hlines(y_key, 0, y_val, color=get_next_color(ax), alpha=alpha, **kwargs)
                    else:
                        ax_marg_y.hist(y, color=get_next_color(ax), normed=True,
                                       alpha=alpha, bins=bins, orientation='horizontal')
                plt.setp(ax_marg_x.get_xticklabels(), visible=False)
                plt.setp(ax_marg_y.get_yticklabels(), visible=False)
            else:
                fig = plt.gcf()
                ax = plt.gca()
                color = get_next_color(ax)

            nullfmt = NullFormatter() #removes labels on fig

            if 'scatter' in type:
                if jitter:
                    x += np.random.normal(loc=0, scale=.01 * (max(x) - min(x)), size=len(x))
                    y += np.random.normal(loc=0, scale=.01 * (max(y) - min(y)), size=len(y))
                ax.scatter(x, y, alpha=alpha, c=color, **kwargs)
            elif 'hist' in type:
                histo = ax.hist2d(x, y, bins=bins, cmap='Blues')
                if 'marginal' not in type:
                    caxes = fig.add_axes([0, 0.1, 0.05, 0.8])
                else:
                    caxes = fig.add_axes([0, 0.1, 0.05, 0.57])
                cbar = plt.colorbar(mappable=histo[3], cax=caxes)
                caxes.yaxis.set_ticks_position('left')
                cbar.set_label('Density')
                caxes.yaxis.set_label_position("left")
                new_labels = []
                for label in caxes.get_yticklabels():
                    new_labels.append(int(label.get_text()) / len(x))
                caxes.set_yticklabels(new_labels)
            elif 'density' in type:
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
                if 'marginal' not in type:
                    caxes = fig.add_axes([0, 0.1, 0.05, 0.8])
                else:
                    caxes = fig.add_axes([0, 0.1, 0.05, 0.57])
                cbar = plt.colorbar(mappable=den, cax=caxes)
                caxes.yaxis.set_ticks_position('left')
                cbar.set_label('Density')
                caxes.yaxis.set_label_position("left")
            elif 'tile' in type:
                #np.unique returns sorted array of unique values
                x_uniq = np.unique(x)
                y_uniq = np.unique(y)
                xmax, xmin = max(x), min(x)
                ymax, ymin = max(y), min(y)
                x_pos = list(range(len(x_uniq)))
                y_pos = list(range(len(y_uniq)))
                x_map = dict(zip(x_uniq, x_pos))
                y_map = dict(zip(y_uniq, y_pos))
                x = np.vectorize(x_map.get)(x)
                y = np.vectorize(y_map.get)(y)
                nums = len(x)

                counts = count_var(list(zip(x, y)))
                intensity = np.zeros(shape=(len(y_pos), len(x_pos)))
                    
                for key, val in counts.items():
                    intensity[key[1]][key[0]] = val / nums
                hm = ax.matshow(intensity, cmap='Blues', origin='lower', aspect='auto')
                ax.xaxis.set_ticks_position('bottom')
                ax.set_xticks(x_pos)
                ax.set_yticks(y_pos)
                ax.set_xticklabels(x_uniq)
                ax.set_yticklabels(y_uniq)

                if 'marginal' not in type:
                    caxes = fig.add_axes([0, 0.1, 0.05, 0.8])
                else:
                    caxes = fig.add_axes([0, 0.1, 0.05, 0.57])
                cbar = plt.colorbar(mappable=hm, cax=caxes)
                caxes.yaxis.set_ticks_position('left')
                cbar.set_label('Relative Frequency')
                caxes.yaxis.set_label_position("left")
            elif 'mixed-tile' in type:
                oxmax, oxmin = max(x), min(x)
                oymax, oymin = max(y), min(y)
                if not discrete_x:
                    x_bin = np.linspace(min(x), max(x), bins)
                    x = np.digitize(x, x_bin)
                elif not discrete_y:
                    y_bin = np.linspace(min(y), max(y), bins)
                    y = np.digitize(y, y_bin)
                #np.unique returns sorted array of unique values
                x_uniq = np.unique(x)
                y_uniq = np.unique(y)
                print(x_uniq)
                xmax, xmin = max(x), min(x)
                ymax, ymin = max(y), min(y)
                x_pos = list(range(len(x_uniq)))
                y_pos = list(range(len(y_uniq)))
                x_map = dict(zip(x_uniq, x_pos))
                y_map = dict(zip(y_uniq, y_pos))
                x = np.vectorize(x_map.get)(x)
                y = np.vectorize(y_map.get)(y)
                nums = len(x)

                counts = count_var(list(zip(x, y)))
                intensity = np.zeros(shape=(len(y_pos), len(x_pos)))
                    
                for key, val in counts.items():
                    intensity[key[1]][key[0]] = val / nums
                hm = ax.matshow(intensity, cmap='Blues', origin='lower', aspect='auto')
                ax.xaxis.set_ticks_position('bottom')
                ax.set_xticks(x_pos)
                ax.set_yticks(y_pos)
                x_lab = np.linspace(oxmin, oxmax, len(x_uniq))
                y_lab = np.linspace(oymin, oymax, len(y_uniq))
                if not discrete_x: x_lab = np.around(x_lab, decimals=1)
                if not discrete_y: y_lab = np.around(y_lab, decimals=1)
                ax.set_xticklabels(x_uniq)
                ax.set_yticklabels(y_uniq)
                #ax.set_xticklabels(x_lab)
                #ax.set_yticklabels(y_lab)

                if 'marginal' not in type:
                    caxes = fig.add_axes([0, 0.1, 0.05, 0.8])
                else:
                    caxes = fig.add_axes([0, 0.1, 0.05, 0.57])
                cbar = plt.colorbar(mappable=hm, cax=caxes)
                caxes.yaxis.set_ticks_position('left')
                cbar.set_label('Relative Frequency')
                caxes.yaxis.set_label_position("left")
            elif 'violin' in type:
                res = np.array(self)
                values = []
                if discrete_x and not discrete_y:
                    positions = sorted(list(x_count.keys()))
                    for i in positions:
                        values.append(res[res[:, 0] == i, 1].tolist())
                    violins = ax.violinplot(dataset=values, showmedians=True)
                    ax.set_xticks(np.array(positions) + 1)
                    ax.set_xticklabels(positions)
                else: 
                    positions = sorted(list(y_count.keys()))
                    for i in positions:
                        values.append(res[res[:, 1] == i, 0].tolist())
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
        else:
            if alpha is None:
                alpha = .1
            for x in self:
                if not hasattr(x, "__iter__"):
                    x = [x]
                plt.plot(x, 'k.-', alpha=alpha, **kwargs)

    def cov(self, **kwargs):
        if get_dimension(self) == 2:
            return np.cov(self, rowvar=False)[0, 1]
        elif get_dimension(self) > 0:
            return np.cov(self, rowvar=False)
        else:
            raise Exception("Covariance requires that the simulation results have consistent dimension.")

    def corr(self, **kwargs):
        if get_dimension(self) == 2:
            return np.corrcoef(self, rowvar=False)[0, 1]
        elif get_dimension(self) > 0:
            return np.corrcoef(self, rowvar=False)
        else:
            raise Exception("Correlation requires that the simulation results have consistent dimension.")

    def mean(self):
        if all(is_scalar(x) for x in self):
            return np.array(self).mean()
        elif get_dimension(self) > 0:
            return tuple(np.array(self).mean(0))
        else:
            raise Exception("I don't know how to take the mean of these values.")

    def var(self):
        if all(is_scalar(x) for x in self):
            return np.array(self).var()
        elif get_dimension(self) > 0:
            return tuple(np.array(self).var(0))
        else:
            raise Exception("I don't know how to take the variance of these values.")

    def sd(self):
        if all(is_scalar(x) for x in self):
            return np.array(self).std()
        elif get_dimension(self) > 0:
            return tuple(np.array(self).std(0))
        else:
            raise Exception("I don't know how to take the variance of these values.")

    def standardize(self):
        mean_ = self.mean()
        sd_ = self.sd() 
        if all(is_scalar(x) for x in self):
            return RVResults((x - mean_) / sd_ for x in self)
        elif get_dimension(self) > 0:
            return RVResults((np.asarray(self) - mean_) / sd_)


class RandomProcessResults(Results):

    def __init__(self, results, timeIndex):
        self.timeIndex = timeIndex
        super().__init__(results)

    def __getitem__(self, t):
        return RVResults(x[t] for x in self)

    def plot(self, tmin=0, tmax=10, alpha=.1, **kwargs):
        if self.timeIndex.fs == float("inf"):
            ts = np.linspace(tmin, tmax, 200)
            style = "k-"
        else:
            nmin = int(np.floor(tmin * self.timeIndex.fs))
            nmax = int(np.ceil(tmax * self.timeIndex.fs))
            ts = [self.timeIndex[n] for n in range(nmin, nmax)]
            style = "k.--"
        for x in self:
            y = [x[t] for t in ts]
            plt.plot(ts, y, style, alpha=alpha, **kwargs)
        plt.xlabel("Time (t)")

        # expand the y-axis slightly
        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        buff = .05 * (ymax - ymin)
        plt.ylim(ymin - buff, ymax + buff)

    def mean(self):
        def fun(t):
            return self[t].mean()
        return TimeFunction(fun, self.timeIndex)

    def var(self):
        def fun(t):
            return self[t].var()
        return TimeFunction(fun, self.timeIndex)

    def sd(self):
        def fun(t):
            return self[t].sd()
        return TimeFunction(fun, self.timeIndex)

