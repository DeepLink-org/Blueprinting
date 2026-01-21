import functools
import html
import sys

import streamlit as st
from htbuilder import span, styles
from htbuilder.units import unit
import hyperparameter as hp
from sympy import Expr, Symbol, latex

from blueprinting.nn.base import TensorDef
from blueprinting.ui import (
    NullFormatter,
    ReadableFLOPs,
    ReadableMem,
    ReadableNum,
    ReadableTime,
)

PALETTE = [
    "#ff4b4b",
    "#ffa421",
    "#ffe312",
    "#21c354",
    "#00d4b1",
    "#00c0f2",
    "#1c83e1",
    "#803df5",
    "#808495",
]

OPACITIES = [
    "33",
    "66",
]

LABEL_SPACING = unit.rem(0.5)
LABEL_FONT_SIZE = unit.rem(0.75)
LABEL_OPACITY = 0.5
LABEL_SPACING = unit.rem(0.5)
PADDING = (unit.rem(0.25), unit.rem(0.5))
BORDER_RADIUS = unit.rem(0.5)

LABEL_DESC = {
    "wgt": "weight mem",
    "act": "activity mem",
    "grad": "grad mem",
    "hbm": "hbm throughput",
    "c2c": "card to card comminications",
}


def _render_html(element, help=None):
    return st.markdown(str(span(element)), unsafe_allow_html=True, help=help)


def labeled_text(label, body, tooltip="-", background=None, color=None, **style):
    long_label = LABEL_DESC.get(label, label)
    color_style = {}

    if color:
        color_style["color"] = color

    if background:
        background_color = background
    else:
        label_sum = sum(ord(c) for c in label)
        background_color = PALETTE[label_sum % len(PALETTE)]
        background_opacity = OPACITIES[label_sum % len(OPACITIES)]
        background = background_color + background_opacity

    separator = (
        span(
            style=styles(
                border_bottom=f"1px solid",
                opacity=0.1,
                margin_bottom=LABEL_SPACING,
                align_self="stretch",
            )
        ),
    )

    label_element = (
        span(
            style=styles(
                margin_bottom=LABEL_SPACING,
                font_size=LABEL_FONT_SIZE,
                opacity=LABEL_OPACITY,
            ),
            title=long_label if long_label else None,
        )(
            html.escape(label),
        ),
        separator,
    )
    return _render_html(
        span(
            style=styles(
                display="inline-flex",
                flex_direction="column",
                align_items="center",
                background=background,
                border_radius=BORDER_RADIUS,
                padding=PADDING,
                overflow="hidden",
                line_height=0.75,
                **color_style,
                **style,
            )
        )(
            label_element,
            html.escape(body),
        ),
        help=tooltip,
    )


def auto_symbol():
    frame = sys._getframe(0).f_back
    subs = {}
    for k, v in frame.f_locals.items():
        if isinstance(v, Symbol):
            subs[v.name] = frame.f_locals[k.lower()]
    return subs


def AnnotatedFormatter(li: TensorDef, subs={}):
    human_readable = hp.scope.blueprinting.formatter.human_readable | True
    mem_fmt = ReadableMem if human_readable else NullFormatter
    num_fmt = ReadableNum if human_readable else NullFormatter
    flops_fmt = ReadableFLOPs if human_readable else NullFormatter
    time_fmt = ReadableTime if human_readable else NullFormatter

    def format(fmt, val, subs):
        if isinstance(val, (Symbol, Expr)):
            expr = latex(val, mul_symbol=" \\times ")
            value = val.subs(subs)
            if value.is_number:
                value = float(value)
            return f"{fmt%value}", f"${expr}$"
        return (f"{fmt%val}",)

    fw, bw, shapes, timming = st.tabs(["forward", "backward", "shapes", "timming"])
    with fw:
        columns = [
            (labeled_text, "wgt", *format(mem_fmt, li.nbytes_weight, subs)),
            (labeled_text, "act", *format(mem_fmt, li.nbytes_activity, subs)),
            (labeled_text, "flops", *format(flops_fmt, li.flops_fw, subs)),
            (labeled_text, "hbm", *format(mem_fmt, li.memory_fw, subs)),
            # (labeled_text, "c2c", *format(flops_fmt, li.c2c_fw, subs)),
        ]
        cs = st.columns(len(columns))
        for col, args in zip(cs, columns):
            with col:
                args[0](*args[1:])
    with bw:
        columns = [
            (labeled_text, "wgt", *format(mem_fmt, li.nbytes_weight, subs)),
            (labeled_text, "act", *format(mem_fmt, li.nbytes_activity, subs)),
            (labeled_text, "grad", *format(mem_fmt, li.nbytes_weight_grads, subs)),
            (labeled_text, "flops", *format(flops_fmt, li.flops_bw, subs)),
            (labeled_text, "hbm", *format(mem_fmt, li.memory_bw, subs)),
            # (labeled_text, "c2c", *format(flops_fmt, li.c2c_bw, subs)),
        ]
        cs = st.columns(len(columns))
        for col, args in zip(cs, columns):
            with col:
                args[0](*args[1:])

    with shapes:
        ins = tuple(x.subs(subs) for x in li.inputs)
        out = li.subs(subs)
        st.markdown(f"${ins} \\rightarrow {out}$")

    with timming:
        with hp.scope(**{"blueprinting.symbolic.subs": subs.items()}):
            columns = [
                (labeled_text, "fw", *format(time_fmt, li.time_fw, subs)),
                (labeled_text, "bw", *format(time_fmt, li.time_bw, subs)),
                # (labeled_text, "c2c", *format(flops_fmt, "#TODO", subs)),
            ]
            cs = st.columns(len(columns))
            for col, args in zip(cs, columns):
                with col:
                    args[0](*args[1:])


AnnotatedFormatter.with_subs = lambda subs: functools.partial(AnnotatedFormatter, subs=subs)

DefaultFormatter = AnnotatedFormatter
