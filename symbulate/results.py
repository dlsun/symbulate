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

def is_hashable(x):
    return x.__hash__ is not None

class Results(list):

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
    row_template = "<tr><td>%s</td><td>%s</td></tr>"

    def __init__(self, results):
        for result in results:
            self.append(result)

    def filter(self, fun):
        return type(self)(x for x in self if fun(x))

    def filter_eq(self, value):
        return self.filter(lambda x: x == value)

    def filter_neq(self, value):
        return self.filter(lambda x: x != value)

    def filter_leq(self, value):
        return self.filter(lambda x: x <= value)

    def filter_geq(self, value):
        return self.filter(lambda x: x >= value)

    def filter_lt(self, value):
        return self.filter(lambda x: x < value)

    def filter_gt(self, value):
        return self.filter(lambda x: x > value)

    def apply(self, fun):
        return type(self)(fun(x) for x in self)

    def component(self, i):
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
        table = Table(self._get_counts())
        if relfreq:
            return table / len(self)
        else:
            return table

    def _repr_html_(self):

        def truncate(result):
            if len(result) > 100:
                return result[:100] + "..."
            else:
                return result

        table_body = ""
        for i, x in enumerate(self):
            table_body += self.row_template % (i, truncate(str(x)))
            # if we've already printed 9 rows, skip to end
            if i >= 8:
                table_body += "<tr><td>...</td><td>...</td></tr>"
                table_body += self.row_template % (len(self) - 1, truncate(str(self[-1])))
                break
        return self.table_template.format(table_body=table_body)


class RVResults(Results):

    def check_single_number(self):
        for x in self:
            if not isinstance(x, Number):
                raise Exception("The output of each simulation must be a single number.")        

    def check_pair(self):
        for x in self:
            if not hasattr(x, "__len__") or len(x) != 2:
                raise Exception("The output of each simulation must be a pair of numbers.")

    def hist(self, type="bar", relfreq=False, xlabel=None, ylabel=None, **kwargs):
        self.check_single_number()
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
        self.check_pair()
        x, y = zip(*self)
        plt.scatter(x, y, **kwargs)

    def mean(self):
        self.check_single_number()
        return np.array(self).mean()

    def var(self):
        self.check_single_number()
        return np.array(self).var()

    def sd(self):
        self.check_single_number()
        return np.array(self).std()
