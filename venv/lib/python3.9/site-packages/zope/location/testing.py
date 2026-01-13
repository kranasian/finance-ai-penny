import importlib
import unittest


def skipUnlessImportable(module: str):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:  # pragma: no cover
        return unittest.skip(f"{module!r} not importable")
    return lambda func: func
