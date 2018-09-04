"""Data structures for storing the resuAlts of a simulation.

This module provides data structures for storing the
results of a simulation, either outcomes from a
probability space or realizations of a random variable /
random process.
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import Affine2D

from .plot import (configure_axes, get_next_color, is_discrete,
    count_var, compute_density, add_colorbar, make_tile,
    setup_ticks, make_violin, make_marginal_impulse, make_density2D)
from .result import (
    Scalar, Vector,
    InfiniteVector, TimeFunction,
    is_scalar, is_vector, is_time_function
)
from .table import Table


plt.style.use('seaborn-colorblind')


def is_hashable(x):
    return x.__hash__ is not None

class Results(object):

    def __init__(self, results):
        self.results = list(results)

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
        return type(self)(fun(x) for x in self.results)

    def __getitem__(self, i):
        return self.apply(lambda x: x[i])

    def __iter__(self):
        for x in self.results:
            yield x

    def __len__(self):
        return len(self.results)

    def get(self, i):
        for j, x in enumerate(self.results):
            if j == i:
                return x

    def _get_counts(self):
        counts = {}
        for x in self.results:
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
        return type(self)(x for x in self.results if fun(x))

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
        for i, x in enumerate(self.results):
            table_body += row_template % (i, truncate(str(x)))
            # if we've already printed 9 rows, skip to end
            if i >= 8:
                table_body += "<tr><td>...</td><td>...</td></tr>"
                i_last = len(self) - 1
                table_body += row_template % (i_last, truncate(str(self.get(i_last))))
                break
        return table_template.format(table_body=table_body)


class RVResults(Results):

    def __init__(self, results):
        super().__init__(results)
        # determine the dimension and the index set (if applicable) of the Results
        self.dim = None
        self.index_set = None
        iterresults = iter(self)
        # get type and dimension of the first result
        first_result = next(iterresults)
        if is_time_function(first_result):
            self.index_set = first_result.index_set
        if is_scalar(first_result):
            self.dim = 1
        elif is_vector(first_result):
            self.dim = len(first_result)
        # iterate over remaining results, ensure they are consistent with the first
        for result in iterresults:
            if (is_time_function(result) and
                result.index_set != self.index_set):
                self.index_set = None
                break
            if ((is_scalar(result) and self.dim != 1) or
                (is_vector(result) and self.dim != len(result))):
                self.dim = None
                break
        # if appropriate, convert results to Numpy arrays
        if self.dim is not None:
            self.array = np.array(self.results)
    
    def plot(self, type=None, alpha=None, normalize=True, jitter=False, 
        bins=None, **kwargs):
        if type is not None:
            if isinstance(type, str):
                type = (type,)
            elif not isinstance(type, (tuple, list)):
                raise Exception("I don't know how to plot a " + str(type))

        # N.B. If self.dim is defined, then self.array is a Numpy array.
        if self.dim == 1:
            # determine plotting parameters
            counts = self._get_counts()
            discrete = is_discrete(counts.values())
            if type is None:
                type = ("impulse", ) if discrete else ("hist", )
            if alpha is None:
                alpha = .5
            if bins is None:
                bins = 30
            n = len(self)

            # initialize figure
            fig = plt.gcf()
            ax = plt.gca()
            color = get_next_color(ax)
            
            if 'density' in type:
                if discrete:
                    xs = sorted(list(counts.keys()))
                    probs = [counts[x] / n for x in xs]
                    ax.plot(xs, probs, marker='o', color=color, linestyle='-')
                    if len(type) == 1:
                        plt.ylabel('Relative Frequency')
                else:
                    density = compute_density(self.array)
                    xs = np.linspace(self.array.min(), self.array.max(), 1000)
                    ax.plot(xs, density(xs), linewidth=2, color=color)
                    if len(type) == 1 or (len(type) == 2 and 'rug' in type):
                        plt.ylabel('Density')

            if 'hist' in type or 'bar' in type:
                ax.hist(self.array, bins=bins, normed=True,
                        color=color, alpha=alpha, **kwargs)
                plt.ylabel("Density" if normalize else "Count")
            elif 'impulse' in type:
                xs = list(counts.keys())
                freqs = list(counts.values())
                if normalize:
                    freqs = [freq / n for freq in freqs]
                if jitter:
                    a = .02 * (max(xs) - min(xs))
                    xs = [x + np.random.uniform(low=-a, high=a) for x in xs]
                # plot the impulses
                ax.vlines(xs, 0, freqs, color=color, alpha=alpha, **kwargs)
                configure_axes(ax, xs, freqs,
                               ylabel="Relative Frequency" if normalize else "Count")
            if 'rug' in type:
                if discrete:
                    noise_level = .002 * (self.array.max() - self.array.min())
                    xs = self.results + np.random.normal(scale=noise_level, size=n)
                ax.plot(xs, [0.001] * n, '|',
                        linewidth = 5, color='k')
                if len(type) == 1:
                    setup_ticks([], [], ax.yaxis)
        elif self.dim == 2:
            x, y = self.array[:, 0], self.array[:, 1]

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
                if discrete_x and not discrete_y:
                    positions = sorted(list(x_count.keys()))
                    make_violin(self.array, positions, ax, 'x', alpha)
                elif not discrete_x and discrete_y: 
                    positions = sorted(list(y_count.keys()))
                    make_violin(self.array, positions, ax, 'y', alpha)
        else:
            if alpha is None:
                alpha = np.log(2) / np.log(len(self) + 1)
            ax = plt.gca()
            color = get_next_color(ax)
            for result in self.results:
                result.plot(alpha=alpha, color=color)
            plt.xlabel("Index")

    def mean(self):
        if self.dim == 1:
            return Scalar(self.array.mean())
        elif self.dim is not None:
            return Vector(self.array.mean(axis=0))
        elif self.index_set is not None:
            def fn(t):
                return self[t].mean()
            return TimeFunction.from_index_set(self.index_set, fn)
        else:
            raise Exception("I don't know how to take the mean of these values.")

    def var(self):
        if self.dim == 1:
            return Scalar(self.array.var())
        elif self.dim is not None:
            return Vector(self.array.var(axis=0))
        elif self.index_set is not None:
            def fn(t):
                return self[t].var()
            return TimeFunction.from_index_set(self.index_set, fn)
        else:
            raise Exception("I don't know how to take the variance of these values.")

    def std(self):
        if self.dim == 1:
            return Scalar(self.array.std())
        elif self.dim is not None:
            return Vector(self.array.std(axis=0))
        elif self.index_set is not None:
            def fn(t):
                return self[t].std()
            return TimeFunction.from_index_set(self.index_set, fn)
        else:
            raise Exception("I don't know how to take the SD of these values.")

    def sd(self):
        return self.std()

    def quantile(self, q):
        if self.dim == 1:
            return Scalar(np.percentile(self.array, q * 100))
        elif self.dim is not None:
            return Vector(np.percentile(self.array, q * 100, axis=0))
        elif self.index_set is not None:
            def fn(t):
                return self[t].quantile(q)
            return TimeFunction.from_index_set(self.index_set, fn)
        else:
            raise Exception("I don't know how to take the quanile of these values.")

    def median(self):
        return self.quantile(.5)
        
    def orderstatistics(self, n):
        if self.dim == 1:
            return Scalar(np.partition(self.array, n - 1)[n - 1])
        elif self.dim is not None:
            return Vector(np.partition(self.array, n - 1, axis=0)[n - 1])
        elif self.index_set is not None:
            def fn(t):
                return self[t].orderstatistics(n)
            return TimeFunction.from_index_set(self.index_set, fn)
        else:                                                                        
            raise Exception("I don't know how to take the order statistics of these values.")
        
    def min(self):
        if self.dim == 1:
            return Scalar(self.array.min())
        elif self.dim is not None:
            return Vector(self.array.min(axis=0))
        elif self.index_set is not None:
            def fn(t):
                return self[t].min()
            return TimeFunction.from_index_set(self.index_set, fn)
        else:
            raise Exception("I don't know how to take the minimum of these values.")
            
    def max(self):
        if self.dim == 1:
            return Scalar(self.array.max())
        elif self.dim is not None:
            return Vector(self.array.max(axis=0))
        elif self.index_set is not None:
            def fn(t):
                return self[t].max()
            return TimeFunction.from_index_set(self.index_set, fn)

    def min_max_diff(self):
        if self.dim == 1:
            return Scalar(self.array.max() - self.array.min())
        elif self.dim is not None:
            return Vector(self.array.max(axis=0) -
                          self.array.min(axis=0))
        elif self.index_set is not None:
            def fn(t):
                return self[t].min_max_diff()
            return TimeFunction.from_index_set(self.index_set, fn)
        else:
            raise Exception("I don't know how to take the range of these values.")

    def iqr(self):
        if self.dim == 1:
            return Scalar(np.percentile(self.array, 75) -
                          np.percentile(self.array, 25))
        elif self.dim is not None:
            return Vector(np.percentile(self.array, 75, axis=0) -
                          np.percentile(self.array, 25, axis=0))
        elif self.index_set is not None:
            def fn(t):
                return self[t].iqr()
            return TimeFunction.from_index_set(self.index_set, fn)
        else:                                                                        
            raise Exception("I don't know how to take the interquartile range of these values.")

    def skewness(self):
        if self.dim == 1:
            return Scalar(stats.skew(self.array))
        elif self.dim is not None:
            return Vector(stats.skew(self.array, axis=0))
        elif self.index_set is not None:
            def fn(t):
                return self[t].skewness()
            return TimeFunction.from_index_set(self.index_set, fn)
        else:
            raise Exception("I don't know how to take the skewness of these values.")

    def kurtosis(self):
        if self.dim == 1:
            return Scalar(stats.kurtosis(self.array))
        elif self.dim is not None:
            return Vector(stats.kurtosis(self.array, axis=0))
        elif self.index_set is not None:
            def fn(t):
                return self[t].kurtosis()
            return TimeFunction.from_index_set(self.index_set, fn)
        else:
            raise Exception("I don't know how to take the kurtosis of these values.")
 
    def moment(self, k):
        if self.dim == 1:
            return Scalar(stats.moment(self.array, k))
        elif self.dim is not None:
            return Vector(stats.moment(self.array, k, axis=0))
        elif self.index_set is not None:
            def fn(t):
                return self[t].moment(k)
            return TimeFunction.from_index_set(self.index_set, fn)
        else:                                                                        
            raise Exception("I don't know how to find the moment of these values.")
    
    def trimmed_mean(self, alpha):
        if self.dim == 1:
            return Scalar(stats.trim_mean(self.array, alpha))
        elif self.dim is not None:
            return Vector(stats.trim_mean(self.array,
                                          alpha, axis=0))
        elif self.index_set is not None:
            def fn(t):
                return self[t].trimmed_mean(alpha)
            return TimeFunction.from_index_set(self.index_set, fn)
        else:                                                                        
            raise Exception("I don't know how to take the trimmed_mean of these values.")
            
    def cov(self, **kwargs):
        if self.dim == 2:
            return np.cov(self.array, rowvar=False)[0, 1]
        elif self.dim > 2:
            return np.cov(self.array, rowvar=False)
        elif self.dim == 1:
            raise Exception("Covariance can only be calculated when there are at least 2 dimensions.")
        else:
            raise Exception("Covariance requires that the simulation results have consistent dimension.")

    def corr(self, **kwargs):
        if self.dim == 2:
            return np.corrcoef(self.array, rowvar=False)[0, 1]
        elif self.dim > 2:
            return np.corrcoef(self.array, rowvar=False)
        elif self.dim == 1:
            raise Exception("Covariance can only be calculated when there are at least 2 dimensions.")
        else:
            raise Exception("Correlation requires that the simulation results have consistent dimension.")

    def standardize(self):
        if self.dim == 1:
            mean_ = self.array.mean()
            sd_ = self.array.std()
            return RVResults((x - mean_) / sd_ for x in self.results)
        elif self.dim > 1:
            mean_ = self.array.mean(axis=0)
            sd_ = self.array.std(axis=0)
            return RVResults((self.results - mean_) / sd_)
        else:
            raise Exception("Could not standardize the given results.")

