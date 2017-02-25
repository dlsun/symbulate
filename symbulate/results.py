"""Data structures for storing the results of a simulation.

This module provides data structures for storing the
results of a simulation, either outcomes from a
probability space or realizations of a random variable /
random process.
"""

import numpy as np
import matplotlib.pyplot as plt

from numbers import Number

from .table import Table
from .utils import is_scalar, is_vector, get_dimension

plt.style.use('ggplot')

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

    ## TODO: Remove .component(i). Use X[i] instead.
    def component(self, i):
        """Returns the ith component of each outcome.
             Used when each outcome consists of several values.

        Args:
          i (int): The component to extract from each outcome.

        Returns:
          Results: A Results object of the same length,
            where each outcome is the ith component of
            an outcome from the original Results object.
        """
        return self.apply(lambda x: x[i])

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

    def tabulate(self, relfreq=False):
        """Counts up how much of each outcome there were.

        Args:
          relfreq (bool): If True, return the relative
            frequency, instead of counts. Defaults to
            False.

        Returns:
          Table: A Table with each of the observed
            outcomes and their freuencies.
        """
        table = Table(self._get_counts())
        if relfreq:
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

    def plot_sample_paths(self, alpha=.1, xlabel=None, ylabel=None, **kwargs):
        for x in self:
            if not hasattr(x, "__iter__"):
                x = [x]
            plt.plot(x, 'k.-', alpha=alpha, **kwargs)
            if xlabel is not None:
                plt.xlabel(xlabel)
            if ylabel is not None:
                plt.ylabel(ylabel)

    def hist(self, type="bar", relfreq=False, xlabel=None, ylabel=None, **kwargs):
        if type == "bar":
            plt.hist(self, alpha=.5, normed=relfreq, **kwargs)
        elif type == "line":
            heights = self._get_counts()
            x = list(heights.keys())
            y = list(heights.values())
            if relfreq:
                y_tot = sum(y)
                y = [i / y_tot for i in y]
            plt.vlines(x, 0, y)
            # create 5% buffer on either end of plot so that leftmost and rightmost lines are visible
            buff = .05 * (max(x) - min(x))
            plt.xlim(min(x) - buff, max(x) + buff)
        else:
            raise Exception("Histogram must have type='line' or 'bar'.")
        # other plot features
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        else:
            ylab = "Relative Frequency" if relfreq else "Count"
            plt.ylabel(ylab)
            
    def scatter(self, **kwargs):
        if get_dimension(self) == 2:
            x, y = zip(*self)
            plt.scatter(x, y, **kwargs)
        else:
            raise Exception("I don't know how to make a scatterplot of more than two variables.")

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


class RandomProcessResults(Results):

    def __init__(self, results, timeIndex):
        self.timeIndex = timeIndex
        super().__init__(results)

    def __getitem__(self, i):
        return Results(x[i] for x in self)

    def plot(self, tmin=0, tmax=10, alpha=.1, **kwargs):
        if self.timeIndex.fs == float("inf"):
            ts = np.linspace(tmin, tmax, 200)
        else:
            nmin = int(np.floor(tmin * self.timeIndex.fs))
            nmax = int(np.ceil(tmax * self.timeIndex.fs))
            ts = [self.timeIndex[n] for n in range(nmin, nmax)]
        for x in self:
            y = [x[t] for t in ts]
            plt.plot(ts, y, 'k-', alpha=alpha, **kwargs)
            plt.xlabel("Time (t)")
