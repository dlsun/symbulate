"""Data structure for storing tabulated results.

This module defines a data structure, Table, that stores the
output of a .tabulate() operation.  Typically, Table stores
the possible outcomes and their counts or relative frequencies.
"""
from .base import Arithmetic


TABLE_TEMPLATE = '''
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


def _get_row_html(outcome, count):
    return "<tr><td>%s</td><td>%s</td></tr>" % (outcome, count)


class Table(dict, Arithmetic):

    def __init__(self, hash_map, outcomes=None):
        self.outcomes = outcomes
        if outcomes is None:
            for outcome, count in hash_map.items():
                self[outcome] = count
        else:
            for outcome in outcomes:
                self[outcome] = (
                    hash_map[outcome] if outcome in hash_map
                    else 0
                )

    def _repr_html_(self):
        # get keys in order
        if self.outcomes is None:
            keys = list(self.keys())
            try:
                keys.sort()
            except Exception:
                pass
        else:
            # preserve ordering of outcomes, if specified
            keys = self.outcomes

        # get HTML for table body
        table_body = ""
        for i, key in enumerate(keys):
            table_body += _get_row_html(key, self[key])
            # if we've already printed 19 rows, skip to end
            if i >= 18:
                table_body += _get_row_html("...", "...")
                table_body += _get_row_html(keys[-1], self[keys[-1]])
                break
        total = str(sum(self.values()))
        table_body += _get_row_html("<b>Total</b>", "<b>%s</b>" % total)

        # return HTML for entire table
        return TABLE_TEMPLATE.format(table_body=table_body)

    # The Arithmetic superclass will use this to define all of the
    # usual arithmetic operations (e.g., +, -, *, /, **, ^, etc.).
    def _operation_factory(self, op):

        def _op_func(self, other):
            return Table(
                {outcome: op(count, other) for outcome, count in self.items()},
                self.outcomes
            )

        return _op_func
