import math

import numpy as np
import scipy.stats as stats

class Arithmetic:
    """A class with operations such as +, -, *, /.

    Subclasses must implement the _operation_factory method,
    which specifies how each operation acts on instances of
    that class.
    """

    # e.g., X + Y or X + 3
    def __add__(self, other):
        op_func = self._operation_factory(lambda x, y: x + y)
        return op_func(self, other)

    # e.g., 3 + X
    def __radd__(self, other):
        return self.__add__(other)

    # e.g., X - Y or X - 3
    def __sub__(self, other):
        op_func = self._operation_factory(lambda x, y: x - y)
        return op_func(self, other)

    # e.g., 3 - X
    def __rsub__(self, other):
        return -1 * self.__sub__(other)

    # e.g., -X
    def __neg__(self):
        return -1 * self

    # e.g., X * Y or X * 2
    def __mul__(self, other):
        op_func = self._operation_factory(lambda x, y: x * y)
        return op_func(self, other)

    # e.g., 2 * X
    def __rmul__(self, other):
        return self.__mul__(other)

    # e.g., X / Y or X / 2
    def __truediv__(self, other):
        op_func = self._operation_factory(lambda x, y: x / y)
        return op_func(self, other)

    # e.g., 2 / X
    def __rtruediv__(self, other):
        op_func = self._operation_factory(lambda x, y: y / x)
        return op_func(self, other)

    # e.g., X ** 2
    def __pow__(self, other):
        op_func = self._operation_factory(lambda x, y: x ** y)
        return op_func(self, other)

    # e.g., 2 ** X
    def __rpow__(self, other):
        op_func = self._operation_factory(lambda x, y: y ** x)
        return op_func(self, other)

    # Alternative notation for powers: e.g., X ^ 2
    def __xor__(self, other):
        return self.__pow__(other)

    # Alternative notation for powers: e.g., 2 ^ X
    def __rxor__(self, other):
        return self.__rpow__(other)


class Comparable:
    """A class with comparison operators such as <, >, and ==.

    Subclasses must implement the _comparison_factory method,
    which specifies how each comparison acts on instances of
    that class.
    """

    def __eq__(self, other):
        op_func = self._comparison_factory(lambda x, y: x == y)
        return op_func(self, other)

    def __ne__(self, other):
        op_func = self._comparison_factory(lambda x, y: x != y)
        return op_func(self, other)

    def __lt__(self, other):
        op_func = self._comparison_factory(lambda x, y: x < y)
        return op_func(self, other)

    def __le__(self, other):
        op_func = self._comparison_factory(lambda x, y: x <= y)
        return op_func(self, other)

    def __gt__(self, other):
        op_func = self._comparison_factory(lambda x, y: x > y)
        return op_func(self, other)

    def __ge__(self, other):
        op_func = self._comparison_factory(lambda x, y: x >= y)
        return op_func(self, other)


class Statistical:
    """A class with statistical functions, such as mean, var, etc.

    Subclasses must implement the _statistic_factory and 
    _multivariate_statistic_factory methods, which specify how
    (univariate) statistics (e.g., mean and variance), as well as
    multivariate statistics (e.g., covariance and correlation)
    are calculated on the object.
    """

    def sum(self):
        r"""Calculate the sum.

        .. math:: \frac{1}{n} \sum_{i=1}^n x_i

        Returns:
          The sum of the numbers.
        """
        op_func = self._statistic_factory(np.sum)
        return op_func(self)

    def mean(self):
        r"""Calculate the mean (a.k.a. average).

        The mean, or average, is a measure of center.

        .. math:: \mu = \frac{1}{n} \sum_{i=1}^n x_i

        Returns:
          float: The mean of the numbers.
        """
        op_func = self._statistic_factory(np.mean)
        return op_func(self)

    def quantile(self, q):
        r"""Calculate a specified quantile (percentile).

        The (100q)th quantile is the value x such that

        .. math:: \frac{\#\{ i: x_i \leq x \}}{n} = q

        Args:
          q (float): A number between 0 and 1 specifying
            the desired quantile or percentile.

        Returns:
          The (100q)th quantile of the numbers.
        """
        op_func = self._statistic_factory(
            lambda **kwargs: np.percentile(q=q * 100, **kwargs)
        )
        return op_func(self)

    def percentile(self, q):
        r"""Calculate a specified percentile.

        Alias for .quantile().
        """
        return self.quantile(q)

    def iqr(self):
        r"""Calculate the interquartile range (IQR).

        The IQR is the 75th percentile minus the 25th percentile.

        Returns:
          The interquartile range.
        """
        return self.quantile(.75) - self.quantile(.25)

    def median(self):
        r"""Calculate the median.

        The median is the middle number in a *sorted* list.
        It is a measure of center.

        Returns:
          The median of the numbers.
        """
        op_func = self._statistic_factory(np.median)
        return op_func(self)

    def std(self):
        r"""Calculate the standard deviation.

        The standard deviation is the square root of the variance.
        It is a measure of spread.

        .. math::

        \sigma &= \sqrt{\frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2} \\
               &= \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2 - \mu^2}

        Returns:
          float: The standard deviation of the numbers.
        """
        op_func = self._statistic_factory(np.std)
        return op_func(self)

    def sd(self):
        r"""Calculate the standard deviation.

        The standard deviation is the square root of the variance.
        It is a measure of spread.

        .. math::

        \sigma &= \sqrt{\frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2} \\
               &= \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2 - \mu^2}

        Returns:
          float: The standard deviation of the numbers.
        """
        return self.std()

    def var(self):
        r"""Calculate the variance.

        The variance is the average squared distance between
        each number and the mean. It is a measure of spread.

        .. math::

        \sigma^2 &= \frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2 \\
               &= \frac{1}{n} \sum_{i=1}^n x_i^2 - \mu^2

        Returns:
          float: The variance of the numbers.
        """
        op_func = self._statistic_factory(np.var)
        return op_func(self)

    def skew(self):
        r"""Calculate the skewness.

        Returns:
          The skewness of the numbers.
        """
        op_func = self._statistic_factory(stats.skew)
        return op_func(self)

    def skewness(self):
        """Calculate the skewness. Alias for .skew()"""
        return self.skew()

    def kurtosis(self):
        r"""Calculate the kurtosis.

        Returns:
          The kurtosis of the numbers.
        """
        op_func = self._statistic_factory(stats.kurtosis)
        return op_func(self)

    def max(self):
        r"""Calculate the maximum.

        Returns:
          The maximum of the numbers.
        """
        op_func = self._statistic_factory(np.amax)
        return op_func(self)

    def min(self):
        r"""Calculate the minimum.

        Returns:
          The minimum of the numbers.
        """
        op_func = self._statistic_factory(np.amin)
        return op_func(self)

    def min_max_diff(self):
        r"""Calculate the difference between the min and max.

        .. math:: \max - \min

        The min-max diff is also called the range. It is
        a measure of spread.

        Returns:
          The difference between the min and the max.
        """
        return self.max() - self.min()

    def cov(self):
        r"""Calculate the pairwise covariances.

        The covariance is a measure of the relationship between two variables.
        The sign of the covariance indicates the direction of the relationship.

        .. math:: 

        \sigma_{XY} = \frac{1}{n} \sum_{i=1}^n (x_i - \mu_X) (y_i - \mu_Y)

        Returns:
          The pairwise covariances between all dimensions. This is usually
          a scalar when there are only 2 dimensions and a matrix when
          there are more than 2 dimensions.
        """
        op_func = self._multivariate_statistic_factory(
            lambda a: np.cov(a, rowvar=False, ddof=0)
        )
        return op_func(self)

    def corr(self):
        r"""Calculate the pairwise correlations.

        The correlation is the covariance normalized by the standard deviations.

        .. math:: 

        \rho_{XY} = \frac{1}{n} \sum_{i=1}^n \frac{x_i - \mu_X}{\sigma_X} \frac{y_i - \mu_Y}{\sigma_Y}

        Returns:
          The pairwise correlations between all dimensions. This is usually
          a scalar when there are only 2 dimensions and a matrix when
          there are more than 2 dimensions.
        """
        op_func = self._multivariate_statistic_factory(
            lambda a: np.corrcoef(a, rowvar=False, ddof=0)
        )
        return op_func(self)

    def corrcoef(self):
        r"""An alias for .corr()"""
        return self.corr()
    

class Logical:
    """A class that supports logical operations: and, or, and not.

    Subclasses must implement the _logical_factory method, which 
    specifies how the logical operator operates on two objects
    of that type.
    """

    def __and__(self, other):
        op_func = self._logical_factory(lambda x, y: x and y)
        return op_func(self, other)

    def __or__(self, other):
        op_func = self._logical_factory(lambda x, y: x or y)
        return op_func(self, other)

    def __invert__(self):
        op_func = self._logical_factory(lambda x: not x)
        return op_func(self)

    
class Filterable:
    """A class with filtering and counting methods.

    Subclasses must implement the filter method, which specifies how to 
    construct a new instance containing only those elements that satisfy 
    a given criterion.
    """

    def filter_eq(self, value):
        """Get all elements equal to a particular value.

        Args:
          value: A value of the same type as the elements in the object.
       
        Returns:
          All of the elements that were equal to value.
        """
        return self.filter(lambda x: x == value)

    def filter_neq(self, value):
        """Get all elements _not_ equal to a particular value.

        Args:
          value: A value of the same type as the elements in the object.
       
        Returns:
          All of the elements that were _not_ equal to value.
        """
        return self.filter(lambda x: x != value)

    def filter_lt(self, value):
        """Get all elements less than a particular value.

        N.B. lt stands for "less than". For elements that are 
        less than _or equal to_ the given value, use .filter_leq(value).

        Args:
          value: A value of the same type as the elements in the object.
       
        Returns:
          All of the elements that were less than value.
        """
        return self.filter(lambda x: x < value)

    def filter_leq(self, value):
        """Get all elements less than or equal to a particular value.

        N.B. leq stands for "less than or equal to". For elements 
        that are strictly less than the given value, use .filter_lt(value).

        Args:
          value: A value of the same type as the elements in the object.
       
        Returns:
          All of the elements that were less than _or equal to_ value.
        """
        return self.filter(lambda x: x <= value)

    def filter_gt(self, value):
        """Get all elements greater than a particular value.

        N.B. gt stands for "greater than". For elements that are 
        greater than _or equal to_ the given value, use .filter_geq(value).

        Args:
          value: A value of the same type as the elements in the object.
       
        Returns:
          All of the elements that were greater than value.
        """

        return self.filter(lambda x: x > value)

    def filter_geq(self, value):
        """Get all elements greater than or equal to a particular value.

        N.B. geq stands for "greater than or equal to". For elements
        that are strictly greater than the given value, use .filter_gt(value).

        Args:
          value: A value of the same type as the elements in the object.
       
        Returns:
          All of the elements that were greater than _or equal to_ value.
        """
        return self.filter(lambda x: x >= value)


    # The following functions return an integer indicating
    # how many elements passed a given criterion.
    
    def count(self, func=lambda x: True):
        """Counts the number of elements satisfying a given criterion.

        Args:
          func (element -> bool): A function that takes in an element
            and returns a boolean (True/False). Only those elements
            that return True will be counted.

        Returns:
          int: The number of elements e for which func(e) is True.
        """
        return len(self.filter(func))

    def count_eq(self, value):
        """Count the number of elements equal to a particular value.

        Args:
          value: A value of the same type as the elements in the object.
       
        Returns:
          int: The number of elements that were equal to value.
        """
        return len(self.filter_eq(value))

    def count_neq(self, value):
        """Count the number of elements _not_ equal to a particular value.

        Args:
          value: A value of the same type as the elements in the object.
       
        Returns:
          int: The number of elements that were not equal to value.
        """
        return len(self.filter_neq(value))

    def count_lt(self, value):
        """Count the number of elements less than a particular value.

        N.B. lt stands for "greater than". For the number of elements
        that are less than _or equal to_ the given value, use 
        .count_leq(value).

        Args:
          value: A value of the same type as the elements in the object.
       
        Returns:
          int: The number of elements that were less than value.
        """
        return len(self.filter_lt(value))

    def count_leq(self, value):
        """Count the number of elements less than or equal to a particular value.

        N.B. leq stands for "less than or equal to". For the number of
        elements that are strictly greater than the given value, use 
        .count_lt(value).

        Args:
          value: A value of the same type as the elements in the object.
       
        Returns:
          int: The number of elements that were less than _or equal to_ value.
        """
        return len(self.filter_leq(value))

    def count_gt(self, value):
        """Count the number of elements greater than a particular value.

        N.B. gt stands for "greater than". For the number of elements
        that are greater than _or equal to_ the given value, use 
        .count_geq(value).

        Args:
          value: A value of the same type as the elements in the object.
       
        Returns:
          int: The number of elements that were greater than value.
        """
        return len(self.filter_gt(value))

    def count_geq(self, value):
        """Count the number of elements greater than or equal to a particular value.

        N.B. geq stands for "greater than or equal to". For the number of
        elements that are strictly greater than the given value, use 
        .count_gt(value).

        Args:
          value: A value of the same type as the elements in the object.
       
        Returns:
          int: The number of elements that were greater than _or equal to_ value.
        """
        return len(self.filter_geq(value))


class Transformable:
    """A class that supports transformations.

    Subclasses must implement the apply method, which specifies how to 
    apply a function to the object.
    """

    def __abs__(self):
        return self.apply(abs)

    def __round__(self):
        return self.apply(round)
    
    def __floor__(self):
        return self.apply(math.floor)

    def __ceil__(self):
        return self.apply(math.ceil)

    
