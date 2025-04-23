import numbers

class LatexTableBuilder:
    def __init__(self, columns, default='-', precision=3):
        """
        columns: list of (display_name, key)
        """
        self.columns = columns
        self.default = default
        self.precision = precision
        self.rows = []

    def round_metric(self, value):
        return round(value, self.precision)

    def add_row(self, system_name, metrics: dict):
        row = [system_name]
        for _, key in self.columns:
            val = metrics.get(key, self.default)
            if isinstance(val, numbers.Number):
                val = self.round_metric(val)
            row.append(str(val))
        self.rows.append(row)

    def render(self):
        # Header
        header = " & ".join(["System"] + [col[0] for col in self.columns]) + " \\\\\n"
        header += "\\\\\n\\midrule\n"

        # Rows
        row_texts = [" & ".join(row) + " \\\\" for row in self.rows]

        return header + "\n".join(row_texts) + "\n"
