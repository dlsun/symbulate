"""Data structure for storing tabulated results.

This module defines a data structure, Table, that stores the
output of a .tabulate() operation.  Typically, Table stores
the possible outcomes and their counts or relative frequencies.
"""
from .base import Arithmetic
from rich.table import Table as RichTable


class Table(dict, Arithmetic):
    def __init__(
        self,
        hash_map,
        outcomes=None,
        normalize=False,
        outcome_column="Outcome",
    ):
        self.outcomes = outcomes
        self.outcome_column = outcome_column
        if outcomes is None:
            for outcome, count in hash_map.items():
                self[outcome] = count
        else:
            for outcome in outcomes:
                self[outcome] = hash_map[outcome] if outcome in hash_map else 0

        if normalize:
            for key in self.ordered_keys():
                self[key] /= sum(hash_map.values())
            self.value_column = "Relative Frequency"
        else:
            self.value_column = "Frequency"

    def ordered_keys(self):
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

        return keys

    def __repr__(self):
        keys = self.ordered_keys()
        keys_strings = [str(x) for x in keys]
        max_key_length = len(max(keys_strings, key=len))

        table_rows = []

        for i, key in enumerate(keys):
            if len(str(key)) <= len(self.outcome_column):
                outcome_space = " " * (len(self.outcome_column) - len(str(key)))
            else:
                outcome_space = " " * (max_key_length - len(str(key)))
            table_rows.append(f"{key}{outcome_space} {self[key]}")

            if i >= 18:
                last_outcome = str(keys[-1])
                last_value = str(self[keys[-1]])
                table_rows.append(
                    f"{'.' * len(last_outcome)}{outcome_space} "
                    f"{'.' * len(last_value)}"
                )
                table_rows.append(f"{last_outcome}{outcome_space} " f"{last_value}")
                break

        if max_key_length <= len(self.outcome_column):
            outcome_header_space = " "
            total_row_space = " " * (len(self.outcome_column) - len("Total"))
        else:
            outcome_header_space = " " * (max_key_length - len(self.outcome_column) + 1)
            total_row_space = " " * (max_key_length - len("Total"))

        total = str(sum(self.values()))
        table_rows.append(f"{total_row_space}Total {total}")
        table_rows.insert(
            0, f"{self.outcome_column}{outcome_header_space}" f"{self.value_column}"
        )

        return "\n".join(table_rows)

    def __rich__(self):
        keys = self.ordered_keys()

        rich_table = RichTable(self.outcome_column, self.value_column)

        num_of_entries = len(keys)
        for i, key in enumerate(keys):
            last_section = i == num_of_entries - 1
            rich_table.add_row(str(key), str(self[key]), end_section=last_section)

        rich_table.add_row("Total", str(sum(self.values())))
        return rich_table

    # The Arithmetic superclass will use this to define all of the
    # usual arithmetic operations (e.g., +, -, *, /, **, ^, etc.).
    def _operation_factory(self, op):
        def _op_func(self, other):
            return Table(
                {outcome: op(count, other) for outcome, count in self.items()},
                self.outcomes,
            )

        return _op_func
