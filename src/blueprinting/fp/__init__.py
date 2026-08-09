import copy


def float_point_values_table(sign_bit=True, exponent_bits=5, mantissa_bits=2, draw=False):
    assert exponent_bits > 0, (
        "Exponent bit amount cannot be zero or negative. A float must have at least 1 exponent bit, or else it's just an integer, loses Inf/NaN, etc."
    )
    assert mantissa_bits >= 0, "Mantissa bit amount cannot be negative. However, mantissa is allowed to have zero bits."
    mantissa_base = _generate_mantissa_base(mantissa_bits)
    table_cell_rows = _generate_table_data_cells(sign_bit, mantissa_base, exponent_bits)
    if draw:
        from IPython.display import Markdown, display

        md = _format_pretty_table(table_cell_rows, sign_bit, exponent_bits, mantissa_bits)
        display(Markdown(md))
    float_point_values = [float(x) for x in sum(table_cell_rows, [])]
    float_point_values = [x for x in float_point_values if x == x and x not in (float("inf"), float("-inf"))]
    float_point_values.sort()
    return float_point_values


def _format_pretty_table(table_cell_rows, sign_bit, exponent_bits, mantissa_bits) -> str:
    nrow = len(table_cell_rows)
    ncol = len(table_cell_rows[0])
    output_text = "| |"
    output_text += "|".join(["… " + _int_to_bits(i, mantissa_bits) for i in range(ncol)])
    output_text += "|\n"
    output_text += "|---|{}|\n".format("|".join(["---"] * ncol))
    for i in range(nrow):
        output_text += "|"
        if sign_bit:
            output_text += "0 " if i < nrow // 2 else "1 "
        output_text += _int_to_bits(i, exponent_bits) + " …|"
        row = table_cell_rows[i]
        output_text += "|".join(row)
        output_text += "|\n"
    return output_text


def _generate_table_data_cells(sign_bit, mantissa_base, exponent_bits):
    rows: list[list] = []
    exponent_bias = int(pow(2, exponent_bits - 1)) - 1
    exponent_range: int = int(pow(2, exponent_bits)) - 1
    # Generate the main rows.
    for i in range(exponent_range):
        row = []
        rows.append(row)
        exponent = i - exponent_bias
        exponent += 1 if i == 0 else 0
        multiplier = pow(2, exponent)
        for j in range(len(mantissa_base)):
            mantissa_number = mantissa_base[j]
            mantissa_number -= 1.0 if i == 0 else 0.0
            stringified_number = str(mantissa_number * multiplier)
            row.append(stringified_number)
    # Add the infinity/NaN row.
    inf_nan_row: list = ["Inf"]
    for _i in range(len(mantissa_base) - 1):
        inf_nan_row.append("NaN")
    rows.append(inf_nan_row)
    # If there is a sign bit, append a duplicate of every row, with a minus sign.
    if sign_bit:
        for row in copy.deepcopy(rows):
            for cell_index in range(len(row)):
                row[cell_index] = "-" + row[cell_index]
            rows.append(row)
    return rows


def _generate_mantissa_base(bits_amount: int):
    base = []
    step = pow(2, -bits_amount)
    value = 1.0
    while True:
        base.append(value)
        value += step
        if value >= 2.0:
            break
    return base


def _int_to_bits(number: int, bits_amount: int) -> str:
    ret: str = ""
    digit_value: int = 1
    for _i in range(bits_amount):
        if number & digit_value == 0:
            ret = "0" + ret
        else:
            ret = "1" + ret
        digit_value *= 2
    return ret
