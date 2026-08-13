"""Project path configuration.

Single source of truth for the directories the analysis code needs:

``data_dir``
    Atlas geometry and regularization kernels. Ships with the repo
    (``./data_public``).
``model_dir``
    Where ``train_rnn.py`` writes trained models. One subdirectory per params
    CSV, e.g. ``<model_dir>/model_params_202606d/``.
``fmri_dir``
    Empirical HCP fMRI inputs. Access-restricted, so these are **not** part of
    the repo -- see the README for how to obtain them.
``figure_dir``
    Where the analysis notebooks write figures. Defaults to ``figures/`` inside
    ``model_dir``, but can point anywhere -- e.g. a synced cloud folder, so
    figures are readable from another device.

Paths are resolved in this order (first hit wins):

1. Environment variables ``NEURO_RNN_DATA_DIR``, ``NEURO_RNN_MODEL_DIR``,
   ``NEURO_RNN_FMRI_DIR``, ``NEURO_RNN_FIGURE_DIR``.
2. A ``paths.yaml`` at the repo root (copy ``paths.yaml.template`` and edit it).
   Its location can be overridden with ``NEURO_RNN_PATHS``.
3. Built-in defaults, for the directories that have sensible ones
   (``data_dir`` -> ``<repo>/data_public``, ``model_dir`` -> ``<repo>/results/model``,
   ``figure_dir`` -> ``<model_dir>/figures``).

Relative paths in ``paths.yaml`` are resolved against the repo root, so the
config behaves the same whether you run from the repo root, from ``scripts/``,
or from inside a notebook.

Typical use::

    from src.config import get_paths, ensure_dir

    paths = get_paths('model_params_202606d', require='all')
    kernels = load_kernels(paths.data_dir)
    figdir = ensure_dir(paths.figure_dir)

Can also be queried from the shell, which is how ``train_rnn.sh`` reads it::

    python -m src.config data_dir
"""

import os
from typing import NamedTuple, Optional

import yaml


# Repo root is the parent of the directory holding this file.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS_TEMPLATE = os.path.join(REPO_ROOT, 'paths.yaml.template')

KEYS = ('data_dir', 'model_dir', 'fmri_dir', 'figure_dir')

ENV_VARS = {
    'data_dir': 'NEURO_RNN_DATA_DIR',
    'model_dir': 'NEURO_RNN_MODEL_DIR',
    'fmri_dir': 'NEURO_RNN_FMRI_DIR',
    'figure_dir': 'NEURO_RNN_FIGURE_DIR',
}

DEFAULTS = {
    'data_dir': os.path.join(REPO_ROOT, 'data_public'),
    'model_dir': os.path.join(REPO_ROOT, 'results', 'model'),
    'fmri_dir': None,    # no sensible default: restricted data, location varies
    'figure_dir': None,  # defaults to <model_dir>/figures; see load_paths
}

# Everything except fmri_dir always resolves to something, so requiring them is
# free. fmri_dir is the only path that can legitimately be unset -- analyses
# that read empirical fMRI should ask for it with ``require='all'``.
DEFAULT_REQUIRE = ('data_dir', 'model_dir', 'figure_dir')


class Paths(NamedTuple):
    """Resolved project directories.

    A NamedTuple so callers can use attribute access (``paths.figure_dir``)
    and so new directories can be added later without breaking existing code.
    """

    data_dir: str
    model_dir: str
    fmri_dir: Optional[str]
    figure_dir: str


def paths_file():
    """Return the path to the active ``paths.yaml`` (may not exist)."""
    return os.environ.get('NEURO_RNN_PATHS', os.path.join(REPO_ROOT, 'paths.yaml'))


def _resolve(path):
    """Expand ``~``/vars and anchor relative paths to the repo root."""
    if path is None:
        return None
    path = os.path.expanduser(os.path.expandvars(str(path)))
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    return os.path.normpath(path)


def _read_yaml():
    """Return the contents of ``paths.yaml`` as a dict (empty if absent)."""
    fname = paths_file()
    if not os.path.isfile(fname):
        return {}
    with open(fname, 'r') as f:
        return yaml.safe_load(f) or {}


def load_paths(require=DEFAULT_REQUIRE):
    """Resolve the configured directories into a dict.

    This is the single resolution point; everything else in this module is a
    thin wrapper over it.

    Args:
        require: keys that must resolve to a non-empty value; raises
            ``FileNotFoundError`` naming the missing key if one does not.
            Pass ``'all'`` to require every key (including ``fmri_dir``).

    Returns:
        dict keyed by :data:`KEYS`, with absolute paths as values
        (``fmri_dir`` may be ``None`` when unset and not required).
    """
    if require == 'all':
        require = KEYS

    from_yaml = _read_yaml()

    paths = {}
    for key in KEYS:
        value = os.environ.get(ENV_VARS[key]) or from_yaml.get(key) or DEFAULTS[key]
        paths[key] = _resolve(value)

    # figure_dir has no static default: it hangs off wherever model_dir landed.
    if not paths['figure_dir']:
        paths['figure_dir'] = os.path.join(paths['model_dir'], 'figures')

    for key in require:
        if not paths.get(key):
            raise FileNotFoundError(
                f"'{key}' is not configured.\n"
                f"Set it in {paths_file()} (copy the template: "
                f"cp {os.path.relpath(PATHS_TEMPLATE, REPO_ROOT)} paths.yaml), "
                f"or export {ENV_VARS[key]}."
            )

    return paths


def get_paths(model_params_name=None, require=DEFAULT_REQUIRE):
    """Return the resolved project directories as a :class:`Paths` tuple.

    When ``model_params_name`` is given it is appended to both ``model_dir``
    and ``figure_dir``, so models and figures from different training sweeps
    are kept apart.

    This resolves paths only -- it does not create anything. Use
    :func:`ensure_dir` on ``figure_dir`` before writing into it.

    Args:
        model_params_name: params-CSV name, e.g. ``'model_params_202606d'``.
        require: see :func:`load_paths`. Analyses that read empirical fMRI
            should pass ``require='all'``.
    """
    paths = load_paths(require=require)

    model_dir = paths['model_dir']
    figure_dir = paths['figure_dir']
    if model_params_name:
        model_dir = os.path.join(model_dir, model_params_name)
        figure_dir = os.path.join(figure_dir, model_params_name)

    return Paths(
        data_dir=paths['data_dir'],
        model_dir=model_dir,
        fmri_dir=paths['fmri_dir'],
        figure_dir=figure_dir,
    )


def ensure_dir(*parts):
    """Create a directory (and parents) if needed and return it.

    Joins its arguments, so a subdirectory can be requested inline::

        figdir = ensure_dir(paths.figure_dir, 'supplementary')
    """
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Print a configured project path.')
    parser.add_argument('key', choices=KEYS, help='which path to print')
    parser.add_argument('--params_name', default=None,
                        help='append this params-CSV name to model_dir/figure_dir')
    args = parser.parse_args()

    print(getattr(get_paths(args.params_name, require=(args.key,)), args.key))
