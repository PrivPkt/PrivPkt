def pick_first_n(gen, n):
    """
    Picks the first N elements of a generator
    """
    idx = 0
    if n > 0:
        for c in gen:
            idx += 1
            yield c
            if idx == n:
                break


def force_length(iterable, n, pad=0):
    """
    Picks the first N of the iterable, returns
    an array and pads the end of it with given
    value if necessary
    :param iterable:
    :param pad:
    :return:
    """
    arr = list(pick_first_n(iterable, n))
    return arr + [pad] * max(0, n - len(arr))


