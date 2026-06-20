"""Package init: enforce Arial as the default font for all figures.

Importing anything from ``src`` sets Arial as the matplotlib default so every
figure (including notebooks that build plots directly with matplotlib/seaborn)
renders with a consistent Arial typeface. Arial is placed first in the
``font.sans-serif`` list so it is also used whenever seaborn resets
``font.family`` to ``'sans-serif'``.
"""
import matplotlib as mpl


def use_arial():
    """Set Arial as the default sans-serif font for matplotlib/seaborn."""
    existing = [f for f in mpl.rcParams['font.sans-serif'] if f != 'Arial']
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial'] + existing


use_arial()
