class Arithmetic:
    """A class with operations such as +, -, *, /.

    Subclasses must implement the _operation_factory method,
    which specifies how each operation acts on instances of
    that class.
    """

    # e.g., X + Y or X + 3
    def __add__(self, other):
        op_fun = self._operation_factory(lambda x, y: x + y)
        return op_fun(self, other)

    # e.g., 3 + X
    def __radd__(self, other):
        return self.__add__(other)

    # e.g., X - Y or X - 3
    def __sub__(self, other):
        op_fun = self._operation_factory(lambda x, y: x - y)
        return op_fun(self, other)

    # e.g., 3 - X
    def __rsub__(self, other):
        return -1 * self.__sub__(other)

    # e.g., -X
    def __neg__(self):
        return -1 * self

    # e.g., X * Y or X * 2
    def __mul__(self, other):
        op_fun = self._operation_factory(lambda x, y: x * y)
        return op_fun(self, other)
            
    # e.g., 2 * X
    def __rmul__(self, other):
        return self.__mul__(other)

    # e.g., X / Y or X / 2
    def __truediv__(self, other):
        op_fun = self._operation_factory(lambda x, y: x / y)
        return op_fun(self, other)

    # e.g., 2 / X
    def __rtruediv__(self, other):
        op_fun = self._operation_factory(lambda x, y: y / x)
        return op_fun(self, other)

    # e.g., X ** 2
    def __pow__(self, other):
        op_fun = self._operation_factory(lambda x, y: x ** y)
        return op_fun(self, other)

    # e.g., 2 ** X
    def __rpow__(self, other):
        op_fun = self._operation_factory(lambda x, y: y ** x)
        return op_fun(self, other)

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
        op_fun = self._comparison_factory(lambda x, y: x == y)
        return op_fun(self, other)

    def __ne__(self, other):
        op_fun = self._comparison_factory(lambda x, y: x != y)
        return op_fun(self, other)

    def __lt__(self, other):
        op_fun = self._comparison_factory(lambda x, y: x < y)
        return op_fun(self, other)

    def __le__(self, other):
        op_fun = self._comparison_factory(lambda x, y: x <= y)
        return op_fun(self, other)

    def __gt__(self, other):
        op_fun = self._comparison_factory(lambda x, y: x > y)
        return op_fun(self, other)

    def __ge__(self, other):
        op_fun = self._comparison_factory(lambda x, y: x >= y)
        return op_fun(self, other)


