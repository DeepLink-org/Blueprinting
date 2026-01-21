import dataclasses


@dataclasses.dataclass
class CommCounter:
    """Counter for communicate operations

    Examples
    --------
    >>> cnt = CommCounter()
    >>> cnt.n_all_reduce
    0

    >>> cnt.n_all_gather += 1
    >>> cnt.n_all_gather
    1
    """

    n_all_reduce: int = 0
    n_all_gather: int = 0
    n_reduce_scatter: int = 0
    n_send_recv: int = 0

    all_reduce: int = 0
    all_gather: int = 0
    reduce_scatter: int = 0
    send_recv: int = 0

    def to_dict(self):
        return {k: v for k, v in dataclasses.asdict(self).items() if v != 0}

    def __str__(self):
        nops = " / ".join(
            str(getattr(self, f.name)) for f in dataclasses.fields(CommCounter) if f.name.startswith("n_")
        )
        ncomm = " / ".join(
            str(getattr(self, f.name)) for f in dataclasses.fields(CommCounter) if not f.name.startswith("n_")
        )
        return f"{nops}\n{ncomm}"

    def __add__(self, other):
        """
        Examples
        --------
        >>> cnt1 = CommCounter(n_all_reduce=1, n_all_gather=2)
        >>> cnt2 = CommCounter(n_all_reduce=3, n_all_gather=4)
        >>> cnt1 + cnt2
        CommCounter(n_all_reduce=4, n_all_gather=6)

        >>> import pandas as pd
        >>> pd.DataFrame([cnt1, cnt2])  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
                n_all_reduce  n_all_gather
        0             1             2
        1             3             4
        >>> pd.DataFrame({"a": [cnt1, cnt2]})["a"].sum()
        CommCounter(n_all_reduce=4, n_all_gather=6)
        """
        cnt = CommCounter()
        for f in dataclasses.fields(CommCounter):
            name = f.name
            setattr(cnt, name, getattr(self, name) + getattr(other, name))
        return cnt

    def __mul__(self, scale):
        cnt = CommCounter()
        for f in dataclasses.fields(CommCounter):
            name = f.name
            setattr(cnt, name, getattr(self, name) * scale)
        return cnt

    __rmul__ = __mul__
