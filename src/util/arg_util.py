import argparse
import pathlib
from os import PathLike


def str2bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        v = v.lower()
        if v in ('t', 'tru', 'true', 'y', 'yes', '1'):
            return True
        elif v in ('f', 'false', 'fals', 'n', 'no', '0'):
            return False
    elif isinstance(v, int):
        return v != 0
    raise argparse.ArgumentTypeError('Boolean value expected.')


def path_abs(v: object) -> pathlib.Path or None:
    if v is None:
        return None
    if isinstance(v, str) or isinstance(v, PathLike):
        return pathlib.Path(v).absolute()
    raise argparse.ArgumentTypeError('str or PathLike value expected.')
