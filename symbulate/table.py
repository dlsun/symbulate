"""Data structure for storing tabulated results.

This module defines a data structure, Table, that stores the
output of a .tabulate() operation.  Typically, Table stores
the possible outcomes and their counts or relative frequencies.
"""
from .base import Arithmetic

class Table(dict, Arithmetic):

    table_template = '''
    <table>
      <thead>
        <th width="80%">Outcome</th>
        <th width="20%">Value</th>
      </thead>
      <tbody>
        {table_body}
      </tbody>
    </table>
    '''

    def _get_row_html(self, key, val):
        return "<tr><td>%s</td><td>%s</td></tr>" % (key, val)

    def __init__(self, hash_map, outcomes=None):
        self.outcomes = outcomes
        if outcomes is None:
            for k, v in hash_map.items():
                self[k] = v
        else:
            for k in outcomes:
                self[k] = hash_map[k] if k in hash_map else 0
    
    def _repr_html_(self):
        # get keys in order
        if self.outcomes is None:
            keys = list(self.keys())
            try:
                keys.sort()
            except:
                pass
        else:
            # preserve ordering of outcomes, if specified
            keys = self.outcomes

        # get HTML for table body
        table_body = ""
        for i, key in enumerate(keys):
            table_body += self._get_row_html(key, self[key])
            # if we've already printed 19 rows, skip to end
            if i >= 18:
                table_body += self._get_row_html("...", "...")
                table_body += self._get_row_html(keys[-1], 
                                                 self[keys[-1]])
                break
        total = str(sum(self.values()))
        table_body += self._get_row_html("<b>Total</b>", 
                                         "<b>" + total + "</b>")

        # return HTML for entire table
        return self.table_template.format(table_body=table_body)

    def _operation_factory(self, op):

        def op_func(self, other):
            return Table(
                {k: op(v, other) for k, v in self.items()},
                self.outcomes
            )

        return op_func
