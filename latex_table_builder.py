import numbers

class LatexTableBuilder:
    def __init__(self, columns, default='\\text{--}', precision=3):
        """
        columns: list of (display_name, key)
        """
        self.columns = columns
        self.default = default
        self.precision = precision
        self.rows = []

    def round_metric(self, value):
        return f"{value:.{self.precision}f}"

    def add_row(self, system_name, metrics: dict):
        row = [system_name]
        for _, key in self.columns:
            val = metrics.get(key, self.default)
            if isinstance(val, dict) and "mean" in val:
                mean_val = val["mean"]
                lower = val.get("lower_ci", "")
                upper = val.get("upper_ci", "")
                mean_val = self.round_metric(mean_val)
                lower = self.round_metric(lower)
                upper = self.round_metric(upper)
                val = f"{mean_val} [{lower}, {upper}]"
            elif isinstance(val, numbers.Number):
                val = self.round_metric(val)
            row.append(str(val))
        self.rows.append(row)

    def render(self):
        # Header
        header = " & " + " & ".join(["System"] + [col[0] for col in self.columns]) + " \\\\\n"
        header += "\\\\\n\\midrule\n"

        # Rows
        row_texts = [" & "  + " & ".join(row) + " \\\\" for row in self.rows]

        return header + "\n".join(row_texts) + "\n"
