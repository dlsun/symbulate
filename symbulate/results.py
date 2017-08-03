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
import seaborn as sns

from numbers import Number

from .sequences import TimeFunction
from .table import Table
from .utils import is_scalar, is_vector, get_dimension
from .plot import configure_axes, get_next_color
from statsmodels.graphics.mosaicplot import mosaic

plt.style.use('ggplot')

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

    def plot(self, type=None, alpha=None, normalize=True, jitter=False, **kwargs):
        dim = get_dimension(self)
        if dim == 1:
            counts = self._get_counts()
            if type is None:
                heights = counts.values()
                if sum([(i > 1) for i in heights]) > .8 * len(heights):
                    type = "impulse"
                else:
                    type = "bar"
            if type == "bar":
                if alpha is None:
                    alpha = .5
                plt.hist(self, normed=normalize, alpha=alpha, **kwargs)
                plt.ylabel("Density" if normalize else "Count")
            elif type == "impulse":
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
                # get next color in cycle
                axes = plt.gca()
                color = get_next_color(axes)
                # plot the impulses
                plt.vlines(x, 0, y, color=color, alpha=alpha, **kwargs)
                
                configure_axes(axes, x, y, ylabel = "Relative Frequency" if normalize else "Count")
            else:
                raise Exception("Histogram must have type='impulse' or 'bar'.")
        elif dim == 2:
            x, y = zip(*self)

            x_count = count_var(x)
            y_count = count_var(y)
            x_height = x_count.values()
            y_height = y_count.values()
            
            discrete_x = False
            if sum([(i > 1) for i in x_height]) > .8 * len(x_height):
                discrete_x = True
            discrete_y = False
            if sum([(i > 1) for i in y_height]) > .8 * len(y_height):
                discrete_y = True

            if type is None:
                if discrete_x and discrete_y:
                    type = "tile"
                elif discrete_x != discrete_y:
                    #TODO will keep scatter for now
                    type = "scatter"
                else:
                    type = "scatter"

            if alpha is None:
                alpha = .5

            if type == "scatter":
                if jitter:
                    x += np.random.normal(loc=0, scale=.01 * (max(x) - min(x)), size=len(x))
                    y += np.random.normal(loc=0, scale=.01 * (max(y) - min(y)), size=len(y))
                # get next color in cycle
                axes = plt.gca()
                color = get_next_color(axes)
                plt.scatter(x, y, color=color, alpha=alpha, **kwargs)
            elif type == "tile" and discrete_x and discrete_y:
                res = pd.DataFrame({'X': x, 'Y': y})
                res['num'] = 1
                temp = pd.pivot_table(res, values = 'num', index = ['Y'],
                    columns = ['X'], aggfunc = np.sum)
                sns.set()
                sns.cubehelix_palette(8)
                fig, ax = plt.subplots(1, 1)
                cbar_ax = fig.add_axes([.91, .3, .03, .4])
                sns.heatmap(temp, ax=ax, cbar_ax = cbar_ax, linewidths = 0.03, 
                    square = True).invert_yaxis()
            elif type == "mosaic" and discrete_x and discrete_y:
                res = pd.DataFrame({'X': x, 'Y': y})
                ct = pd.crosstab(res['Y'], res['X'])
                ctplus = ct + 1e-8
                labels = lambda k: ""
                fig, ax = plt.subplots(1, 1)
                mosaic(ctplus.unstack(), ax = ax, labelizer = labels, axes_label = False)
            elif discrete_x and discrete_y:
                raise Exception("Must have type='mosaic', 'tile', or 'scatter' if discrete.")
            elif (discrete_x and not discrete_y
                or discrete_y and not discrete_x) and type == 'violin':
                #TODO
                raise NotImplementedError
            else:
                raise Exception("Can only have type='scatter' if continuous.")
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
        temp_mean = self.mean()
        temp_sd  = self.sd() 
        if all(is_scalar(x) for x in self):
            return RVResults((x - temp_mean) / temp_sd for x in self)
        elif get_dimension(self) > 0:
            return RVResults((np.asarray(self) - temp_mean) / temp_sd)


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

