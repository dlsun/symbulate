"""Data structures for storing the results of a simulation.

This module provides data structures for storing the
results of a simulation, either outcomes from a
probability space or realizations of a random variable /
random process.
"""
import time

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import Affine2D

from .base import (Arithmetic, Statistical, Comparable,
                   Logical, Filterable, Transformable)
from .plot import (configure_axes, get_next_color, is_discrete,
                   count_var, compute_density, add_colorbar,
                   setup_ticks, make_tile, make_violin,
                   make_marginal_impulse, make_density2D)
from .result import (Scalar, Vector, TimeFunction,
                     is_number, is_numeric_vector)
from .table import Table


plt.style.use('seaborn-colorblind')


def _is_hashable(obj):
    return hasattr(obj, "__hash__")

def _is_boolean_vector(vector):
    return all(isinstance(x, (bool, np.bool_)) for x in vector)


class Results(Arithmetic, Statistical, Comparable,
              Logical, Filterable, Transformable):

    def __init__(self, results, sim_id=None):
        self.results = list(results)
        self.sim_id = time.time() if sim_id is None else sim_id

    def apply(self, func):
        """Apply a function to each outcome of a simulation.

        Args:
          func: A function to apply to each outcome.

        Returns:
          Results: A Results object of the same length,
            where each outcome is the result of applying
            the function to each outcome from the original
            Results object.
        """
        return type(self)(
            [func(result) for result in self.results],
            self.sim_id
        )

    def __getitem__(self, n):
        # if n is a Results object, use it as a boolean mask
        if isinstance(n, Results):
            return self.filter(n)
        # if n is a numeric array of values, return a Results
        # object with those dimensions
        elif is_numeric_vector(n):
            return self.apply(
                lambda result: type(result)(result[i] for i in n)
            )
        # otherwise, return the nth value of every simulation
        return self.apply(lambda result: result[n])

    def __iter__(self):
        for result in self.results:
            yield result

    def __len__(self):
        return len(self.results)

    def get(self, n):
        """Get the outcome of the nth simulation.

        Suppose x is an instance of a Results object.
        Although x behaves like a list, in that you
        can iterate over it, x[n] does not return
        the nth simulation. Instead, it returns a
        Results object, containing the nth dimension
        of every simulation. To get the outcome of the
        nth simulation, the .get(n) method is provided.

        Args:
          n (int): the index of the simulation result to get

        Returns:
          The outcome of the nth simulation.
        """

        # if n is a numeric array, return a Results object with those results
        if is_numeric_vector(n):
            return type(self)(
                self.results[i] for i in n
            )
        # otherwise, return the nth result (this also works when n is a slice)
        return self.results[n]

    def _get_counts(self):
        counts = {}
        for result in self.results:
            if _is_hashable(result):
                outcome = result
            elif isinstance(result, list) and all(_is_hashable(x) for x in result):
                outcome = tuple(result)
            else:
                outcome = str(result)
            if outcome in counts:
                counts[outcome] += 1
            else:
                counts[outcome] = 1
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
            table /= len(self)
        return table

    # The Filterable superclass will use this to define all of the
    # .filter_*() and .count_*() methods.
    def filter(self, filt):
        """Get only the results that satisfy the given criterion.

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
                    "In order to filter one Results object "
                    "by another, they must come from the "
                    "same simulation."
                )
            if len(filt) != len(self):
                raise ValueError(
                    "Filter must be the same length as the "
                    "Results object."
                )
            if not _is_boolean_vector(filt):
                raise ValueError(
                    "Every element in the filter must be a boolean."
                )
            return type(self)(x for x, cond in zip(self, filt) if cond)
        elif callable(filt):
            return type(self)(x for x in self if filt(x))
        else:
            raise TypeError(
                "A filter must be either a function or a "
                "boolean Results object of the same length."
            )

    # The Arithmetic superclass will use this to define all of the
    # usual arithmetic operations (e.g., +, -, *, /, **, ^, etc.).
    def _operation_factory(self, op):

        def _op_func(self, other):
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

        return _op_func

    # The Comparison superclass will use this to define all of the
    # usual comparison operations (e.g., <, >, ==, !=, etc.).
    def _comparison_factory(self, op):
        return self._operation_factory(op)

    # The Statistical superclass will use this to define all of the
    # usual comparison operations (e.g., <, >, ==, !=, etc.).
    def _statistic_factory(self, _):
        raise Exception("Statistical functions are only available "
                        "for simulations of random variables. "
                        "Define a RV on this probability space "
                        "and then try again.")

    def _multivariate_statistic_factory(self, op):
        self._statistic_factory(op)

    # The Logical superclass will use this to define the three
    # logical operations: and (&), or (|), not (~).
    def _logical_factory(self, op):

        def _op_func(self, other=None):
            # check that the vector only contains booleans
            if not _is_boolean_vector(self):
                raise ValueError(
                    "Logical operations are only defined for "
                    "boolean (True/False) Results objects.")
            # other will be None when op is the "not" operator
            if other is None:
                return Results([op(x) for x in self], self.sim_id)
            else:
                if isinstance(other, Results):
                    if self.sim_id != other.sim_id:
                        raise Exception("Results objects must come "
                                        "from the same simulation.")
                    if not _is_boolean_vector(other):
                        raise ValueError(
                            "Logical operations are only defined for "
                            "boolean (True/False) Results objects.")
                else:
                    raise TypeError(
                        "Logical operations are only defined "
                        "between two Results, not between a Result "
                        "and a %s." % type(other).__name__)
                return Results(
                    [op(x, y) for x, y in zip(self, other)],
                    self.sim_id
                )

        return _op_func


    def plot(self):
        raise Exception("Only simulations of random variables (RV) "
                        "can be plotted, but you simulated from a "
                        "probability space. You must first define a RV "
                        "on your probability space and simulate it. "
                        "Then call .plot() on those simulations.")

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

        def _truncate(result):
            if len(result) > 100:
                return result[:100] + "..."
            return result

        table_body = ""
        for i, result in enumerate(self.results):
            table_body += row_template % (i, _truncate(str(result)))
            # if we've already printed 9 rows, skip to end
            if i >= 8:
                table_body += "<tr><td>...</td><td>...</td></tr>"
                i_last = len(self) - 1
                table_body += row_template % (i_last, _truncate(str(self.get(i_last))))
                break
        return table_template.format(table_body=table_body)


class RVResults(Results):

    def __init__(self, results, sim_id=None):
        super().__init__(results, sim_id)
        # get type and dimension of the first result, if it exists
        iterresults = iter(self)
        try:
            first_result = next(iterresults)
        except StopIteration:
            return
        # determine the index set (if each realization is a TimeFunction)
        if isinstance(first_result, TimeFunction):
            self.index_set = first_result.index_set
        else:
            self.index_set = None
        # determine the dimension
        if is_number(first_result):
            self.dim = 1
        elif is_numeric_vector(first_result):
            self.dim = len(first_result)
        else:
            self.dim = None
        # iterate over remaining results, ensure they are consistent with the first
        for result in iterresults:
            if (isinstance(result, TimeFunction) and
                result.index_set != self.index_set):
                self.index_set = None
            if ((is_number(result) and self.dim != 1) or
                (is_numeric_vector(result) and self.dim != len(result))):
                self.dim = None

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

    # The Statistical superclass will use this to define all of the
    # usual comparison operations (e.g., <, >, ==, !=, etc.).
    def _statistic_factory(self, op):

        def _op_func(self):
            self._set_array()
            if self.dim == 1:
                return Scalar(op(a=self.array))
            elif self.dim is not None:
                return Vector(op(a=self.array, axis=0))
            elif self.index_set is not None:
                def _func(t):
                    return _op_func(self[t])
                return TimeFunction.from_index_set(self.index_set, _func)
            raise NotImplementedError(
                "Statistics can only be calculated for numerical "
                "data of consistent dimension."
            )

        return _op_func

    def _multivariate_statistic_factory(self, op):

        def _op_func(self):
            self._set_array()
            if self.dim == 2:
                return op(self.array)[0, 1]
            elif self.dim > 2:
                return op(self.array)
            elif self.dim == 1:
                raise Exception(
                    "This multivariate statistic is only defined when "
                    "when there are at least 2 dimensions.")
            raise NotImplementedError(
                "Statistics can only be calculated for numerical "
                "data of consistent dimension."
            )

        return _op_func

    def standardize(self):
        """Standardizes the results with respect to the mean and standard deviation.

        Returns:
          A new RVResults object, where every dimension has mean 0 and variance 1.
        """
        self._set_array()
        if self.dim is not None:
            return (self - self.mean()) / self.std()
        else:
            raise Exception("Could not standardize the given results.")

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
                ax.plot(xs, [0.001] * n, '|', linewidth=5, color='k')
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
                    ax_marg_x.plot(x_lines, densityX(x_lines), linewidth=2,
                                   color=get_next_color(ax))
                    ax_marg_y.plot(y_lines, densityY(y_lines), linewidth=2,
                                   color=get_next_color(ax),
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
