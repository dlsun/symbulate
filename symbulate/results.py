"""Data structures for storing the results of a simulation.

This module provides data structures for storing the
results of a simulation, either outcomes from a
probability space or realizations of a random variable /
random process.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import Affine2D

from .plot import (configure_axes, get_next_color, is_discrete,
    count_var, compute_density, add_colorbar, make_tile,
    setup_ticks, make_violin, make_marginal_impulse, make_density2D)
from .result import TimeFunction
from .table import Table
from .utils import is_scalar, get_dimension

plt.style.use('seaborn-colorblind')

def is_hashable(x):
    return x.__hash__ is not None

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
            takes in an outcome and returns
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
            discrete = is_discrete(heights)
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
                    density = compute_density(self)
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
                    setup_ticks([], [], ax.yaxis)
        elif dim == 2:
            x, y = zip(*self)

            x_count = count_var(x)
            y_count = count_var(y)
            x_height = x_count.values()
            y_height = y_count.values()
            discrete_x = is_discrete(x_height)
            discrete_y = is_discrete(y_height)

            if type is None:
                type = ("scatter",)
            if alpha is None:
                alpha = .5
            if bins is None:
                bins = 10 if 'tile' in type else 30

            if 'marginal' in type:
                fig = plt.gcf()
                gs = GridSpec(4, 4)
                ax = fig.add_subplot(gs[1:4, 0:3])
                ax_marg_x = fig.add_subplot(gs[0, 0:3])
                ax_marg_y = fig.add_subplot(gs[1:4, 3])
                color = get_next_color(ax)
                if 'density' in type:
                    densityX = compute_density(x)
                    densityY = compute_density(y)
                    x_lines = np.linspace(min(x), max(x), 1000)
                    y_lines = np.linspace(min(y), max(y), 1000)
                    ax_marg_x.plot(x_lines, densityX(x_lines), linewidth=2, color=get_next_color(ax))
                    ax_marg_y.plot(y_lines, densityY(y_lines), linewidth=2, color=get_next_color(ax), 
                                  transform=Affine2D().rotate_deg(270) + ax_marg_y.transData)
                else:
                    if discrete_x:
                        make_marginal_impulse(x_count, get_next_color(ax), ax_marg_x, alpha, 'x')
                    else:
                        ax_marg_x.hist(x, color=get_next_color(ax), normed=True, 
                                       alpha=alpha, bins=bins)
                    if discrete_y:
                        make_marginal_impulse(y_count, get_next_color(ax), ax_marg_y, alpha, 'y')
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
                caxes = add_colorbar(fig, type, histo[3], 'Density')
                #change scale to density instead of counts
                new_labels = []
                for label in caxes.get_yticklabels():
                    new_labels.append(int(label.get_text()) / len(x))
                caxes.set_yticklabels(new_labels)
            elif 'density' in type:
                den = make_density2D(x, y, ax)
                add_colorbar(fig, type, den, 'Density')
            elif 'tile' in type:
                hm = make_tile(x, y, bins, discrete_x, discrete_y, ax)
                add_colorbar(fig, type, hm, 'Relative Frequency')
            elif 'violin' in type:
                res = np.array(self)
                if discrete_x and not discrete_y:
                    positions = sorted(list(x_count.keys()))
                    make_violin(res, positions, ax, 'x', alpha)
                elif not discrete_x and discrete_y: 
                    positions = sorted(list(y_count.keys()))
                    make_violin(res, positions, ax, 'y', alpha)
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
            raise Exception("I don't know how to take the SD of these values.")

    def standardize(self):
        mean_ = self.mean()
        sd_ = self.sd()
        if all(is_scalar(x) for x in self):
            return RVResults((x - mean_) / sd_ for x in self)
        elif get_dimension(self) > 0:
            return RVResults((np.asarray(self) - mean_) / sd_)


class RandomProcessResults(Results):

    def __init__(self, results, index_set):
        self.index_set = index_set
        super().__init__(results)

    def __getitem__(self, t):
        return RVResults(x[t] for x in self)

    def plot(self, tmin=0, tmax=10, **kwargs):
        ax = plt.gca()
        alpha = np.log(2) / np.log(len(self) + 1)
        color = get_next_color(ax)
        for result in self:
            result.plot(tmin, tmax, alpha=alpha, color=color)
        plt.xlabel("Time (t)")

        # expand the y-axis slightly
        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        buff = .05 * (ymax - ymin)
        plt.ylim(ymin - buff, ymax + buff)

    def mean(self):
        def fn(t):
            return self[t].mean()
        return TimeFunction.from_index_set(self.index_set, fn)

    def var(self):
        def fn(t):
            return self[t].var()
        return TimeFunction.from_index_set(self.index_set, fn)

    def sd(self):
        def fn(t):
            return self[t].sd()
        return TimeFunction.from_index_set(self.index_set, fn)

