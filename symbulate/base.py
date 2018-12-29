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

    Subclasses must implement the _statistic_factory method,
    which specifies how the statistic is calculated on the object.
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

        
