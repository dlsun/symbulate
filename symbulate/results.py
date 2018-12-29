"""Data structures for storing the results of a simulation.

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
from time import time

from .base import Arithmetic, Comparable
from .plot import (configure_axes, get_next_color, is_discrete,
    count_var, compute_density, add_colorbar, make_tile,
    setup_ticks, make_violin, make_marginal_impulse, make_density2D)
from .result import (Scalar, Vector, TimeFunction,
                     is_scalar, is_vector)
from .table import Table


plt.style.use('seaborn-colorblind')


def is_hashable(x):
    return hasattr(x, "__hash__")


class Results(Arithmetic, Comparable):

    def __init__(self, results, sim_id=None):
        self.results = list(results)
        self.sim_id = time() if sim_id is None else sim_id

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
        return type(self)(
            [fun(x) for x in self.results],
            self.sim_id
        )

    def __getitem__(self, i):
        if isinstance(i, Results):
            return self.filter(i)
        else:
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

    def filter(self, filt):
        """Filters the results of a simulation and
             returns only those outcomes that satisfy
             a given criterion.

        Args:
          filt: Either a function that takes in
            an outcome and returns a boolean, or
            a Results object of booleans of the 
            same length as this Results object.            

        Returns:
          Results: Another Results object containing
            only those outcomes corresponding to True.
        """
        if isinstance(filt, Results):
            if self.sim_id != filt.sim_id:
                raise Exception(
                    "Results objects must come from the "
                    "same simulation."
                )
            if len(filt) != len(self):
                raise ValueError(
                    "Filter must be the same length "
                    "as the Results object."
                    )
            if not all(type(x) in (bool, np.bool_) for x in filt):
                raise ValueError(
                    "Every element in the filter must "
                    "be a boolean."
                    )
            return type(self)(x for x, y in zip(self, filt) if y)
        elif callable(filt):
            return type(self)(x for x in self.results if filt(x))
        else:
            raise TypeError(
                "A filter must be either a function or a "
                "boolean Results object of the same length."
            )

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

    # e.g., abs(X)
    def __abs__(self):
        return self.apply(abs)

    # The Arithmetic superclass will use this to define all of the
    # usual arithmetic operations (e.g., +, -, *, /, **, ^, etc.).
    def _operation_factory(self, op):

        def op_func(self, other):
            if isinstance(other, Results):
                if len(self) != len(other):
                    raise Exception(
                        "Results objects must be of the "
                        "same length."
                    )
                if self.sim_id != other.sim_id:
                    raise Exception(
                        "Results objects must come from the "
                        "same simulation."
                    )
                return type(self)(
                    [op(x, y) for x, y in zip(self, other)],
                    self.sim_id
                )
            else:
                return self.apply(lambda x: op(x, other))

        return op_func

    # The Comparison superclass will use this to define all of the
    # usual comparison operations (e.g., <, >, ==, !=, etc.).
    def _comparison_factory(self, op):
        return self._operation_factory(op)
    
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

    def __init__(self, results, sim_id=None):
        super().__init__(results, sim_id)
        # determine the dimension and the index set (if applicable) of the Results
        self.dim = None
        self.index_set = None
        # get type and dimension of the first result, if it exists
        iterresults = iter(self)
        try:
            first_result = next(iterresults)
        except:
            return
        if isinstance(first_result, TimeFunction):
            self.index_set = first_result.index_set
        if is_scalar(first_result):
            self.dim = 1
        elif is_vector(first_result):
            self.dim = len(first_result)
        # iterate over remaining results, ensure they are consistent with the first
        for result in iterresults:
            if (isinstance(result, TimeFunction) and
                result.index_set != self.index_set):
                self.index_set = None
                break
            if ((is_scalar(result) and self.dim != 1) or
                (is_vector(result) and self.dim != len(result))):
                self.dim = None
                break

    def _set_array(self):
        # check if it has already been set
        if hasattr(self, "array"):
            return
        # don't set array for TimeFunctions
        elif self.index_set is not None:
            return
        # otherwise set array
        elif self.dim is not None:
            self.array = np.asarray(self.results)
        else:
            raise Exception(
                "This operation is only possible with results "
                "of consistent dimension.")
            
    def plot(self, type=None, alpha=None, normalize=True, jitter=False, 
        bins=None, **kwargs):
        if type is not None:
            if isinstance(type, str):
                type = (type,)
            elif not isinstance(type, (tuple, list)):
                raise Exception("I don't know how to plot a " + str(type))
        
        if self.dim == 1:
            # make sure self.array, a Numpy array, has been set
            self._set_array()
            
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
                ax.hist(self.array, bins=bins, density=normalize,
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
                xs = self.array
                if discrete:
                    noise_level = .002 * (self.array.max() - self.array.min())
                    xs = xs + np.random.normal(scale=noise_level, size=n)
                ax.plot(xs, [0.001] * n, '|', linewidth = 5, color='k')
                if len(type) == 1:
                    setup_ticks([], [], ax.yaxis)
        elif self.dim == 2:
            # make sure self.array, a Numpy array, has been set
            self._set_array()
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
                        ax_marg_x.hist(x, color=get_next_color(ax), density=normalize, 
                                       alpha=alpha, bins=bins)
                    if discrete_y:
                        make_marginal_impulse(y_count, get_next_color(ax), ax_marg_y, alpha, 'y')
                    else:
                        ax_marg_y.hist(y, color=get_next_color(ax), density=normalize,
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
                    x = x + np.random.normal(loc=0, scale=.01 * (x.max() - x.min()), size=len(x))
                    y = y + np.random.normal(loc=0, scale=.01 * (y.max() - y.min()), size=len(y))
                ax.scatter(x, y, alpha=alpha, c=color, **kwargs)
            elif 'hist' in type:
                histo = ax.hist2d(x, y, bins=bins, cmap='Blues')

                # When normalize=True, use density instead of counts
                if normalize:
                    caxes = add_colorbar(fig, type, histo[3], 'Density')
                    #change scale to density instead of counts
                    new_labels = []
                    for label in caxes.get_yticklabels():
                        new_labels.append(int(label.get_text()) / len(x))
                    caxes.set_yticklabels(new_labels)
                else:
                    caxes = add_colorbar(fig, type, histo[3], 'Count')    
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
                result.plot(alpha=alpha, color=color, **kwargs)
            plt.xlabel("Index")
            
    def mean(self):
        self._set_array()
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
        self._set_array()
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
        self._set_array()
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
        self._set_array()
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
        self._set_array()
        return self.quantile(.5)
        
    def orderstatistics(self, n):
        self._set_array()
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
        self._set_array()
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
        self._set_array()
        if self.dim == 1:
            return Scalar(self.array.max())
        elif self.dim is not None:
            return Vector(self.array.max(axis=0))
        elif self.index_set is not None:
            def fn(t):
                return self[t].max()
            return TimeFunction.from_index_set(self.index_set, fn)

    def min_max_diff(self):
        self._set_array()
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
        self._set_array()
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
        self._set_array()
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
        self._set_array()
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
        self._set_array()
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
        self._set_array()
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
        self._set_array()
        if self.dim == 2:
            return np.cov(self.array, rowvar=False)[0, 1]
        elif self.dim > 2:
            return np.cov(self.array, rowvar=False)
        elif self.dim == 1:
            raise Exception("Covariance can only be calculated when there are at least 2 dimensions.")
        else:
            raise Exception("Covariance requires that the simulation results have consistent dimension.")

    def corr(self, **kwargs):
        self._set_array()
        if self.dim == 2:
            return np.corrcoef(self.array, rowvar=False)[0, 1]
        elif self.dim > 2:
            return np.corrcoef(self.array, rowvar=False)
        elif self.dim == 1:
            raise Exception("Covariance can only be calculated when there are at least 2 dimensions.")
        else:
            raise Exception("Correlation requires that the simulation results have consistent dimension.")

    def standardize(self):
        self._set_array()
        if self.dim is not None:
            return (self - self.mean()) / self.std()
        else:
            raise Exception("Could not standardize the given results.")

