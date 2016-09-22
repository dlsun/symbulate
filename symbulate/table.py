"""Data structure for storing tabulated results.

This module defines a data structure, Table, that stores the
output of a .tabulate() operation.  Typically, Table stores
the possible outcomes and their counts or relative frequencies.
"""


class Table(dict):

    table_template = '''
    <table>
      <thead>
        <th width="80%">Result</th>
        <th width="20%">Count</th>
      </thead>
      <tbody>
        {table_body}
      </tbody>
    </table>
    '''
    row_template = "<tr><td>%s</td><td>%s</td></tr>"

    def __init__(self, hash_map):
        for k, v in hash_map.items():
            self[k] = v
    
    def _repr_html_(self):
        kv_list = list(self.items())
        try:
            kv_list.sort(key=lambda kv: kv[0])
        except:
            pass
        table_body = ""
        for i, (key, val) in enumerate(kv_list):
            table_body += self.row_template % (str(key), str(val))
            # if we've already printed 19 rows, skip to end
            if i >= 18:
                table_body += "<tr><td>...</td><td>...</td></tr>"
                last_key, last_val = kv_list[-1]
                table_body += self.row_template % (str(last_key), str(last_val))
                break
        return self.table_template.format(table_body=table_body)

    def _transform_values(self, f):
        return Table({k: f(v) for k, v in self.items()})

    def __add__(self, n):
        return self._transform_values(lambda v: v + n)

    def __radd__(self, n):
        return self.__add__(n)

    def __sub__(self, n):
        return self._transform_values(lambda v: v - n)

    def __rsub__(self, n):
        return self._transform_values(lambda v: n - v)

    def __mul__(self, n):
        return self._transform_values(lambda v: v * n)

    def __rmul__(self, n):
        return self.__mul__(n)
    
    def __truediv__(self, n):
        return self._transform_values(lambda v: v / n)

    def __rtruediv__(self, n):
        return self._transform_values(lambda v: n / v)

    def __pow__(self, n):
        return self._transform_values(lambda v: v ** n)

    def __rpow__(self, n):
        return self._transform_values(lambda v: n ** v)
