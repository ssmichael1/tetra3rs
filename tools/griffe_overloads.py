"""
Griffe extension for the API docs: document functions that exist in the
stub only as ``@overload`` signatures.

A stub (``.pyi``) function with overloads has no implementation, and griffe
attaches overloads only to an implementation, so every overload-only
function (``SolveResult.pixel_to_world`` / ``world_to_pixel``,
``SolverDatabase.calibrate_camera``) was silently left out of the API pages.

For each such name this adds a function built from the overload with the
longest docstring (the general form, where there is one) and lists every
overload under it; ``overloads_only`` in ``mkdocs.yml`` then shows just
the overload signatures.

Enabled in ``mkdocs.yml`` as ``tools/griffe_overloads.py:OverloadOnlyFunctions``.
Same extension as satkit's ``tools/griffe_overloads.py``.
"""

from __future__ import annotations

import ast
from typing import Any

import griffe


class OverloadOnlyFunctions(griffe.Extension):
    def on_members(self, *, node: ast.AST | griffe.ObjectNode, obj: griffe.Object, **kwargs: Any) -> None:
        if obj.kind not in (griffe.Kind.MODULE, griffe.Kind.CLASS):
            return
        for name, overloads in list(obj.overloads.items()):
            if not overloads or name in obj.members:
                continue
            main = max(overloads, key=lambda f: len(f.docstring.value) if f.docstring else 0)
            func = griffe.Function(
                name,
                lineno=main.lineno,
                endlineno=main.endlineno,
                parameters=main.parameters,
                returns=main.returns,
                decorators=[d for d in main.decorators if "overload" not in d.callable_path],
                docstring=main.docstring,
                parent=obj,
            )
            func.overloads = overloads
            obj.set_member(name, func)
            del obj.overloads[name]
