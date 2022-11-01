# -----------------------------------------------------------------------------
# This file was autogenerated by symforce from template:
#     function/namespace_init.py.jinja
# Do NOT modify by hand.
# -----------------------------------------------------------------------------

import importlib
import pkgutil
import sys
import typing as T
from pathlib import Path

if T.TYPE_CHECKING:
    from ._init import *

for finder in sys.meta_path:
    spec = finder.find_spec("codegen_test", path=None)
    if spec is not None:
        __path__.extend(spec.submodule_search_locations)  # type: ignore  # mypy issue #1422
__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore  # mypy issue #1422
seen_paths = set()
for path in __path__:
    resolved = Path(path).resolve()
    if resolved in seen_paths:
        continue
    seen_paths.add(resolved)
    init = resolved / "_init.py"
    if init.is_file():
        spec = importlib.util.spec_from_file_location("codegen_test._init", init)  # type: ignore[attr-defined]
        if spec is not None and spec.loader is not None:
            exec(spec.loader.get_code("codegen_test._init"), globals())  # type: ignore[attr-defined]
