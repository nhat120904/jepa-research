"""Minimal undefined-name checker for function scopes.

Catches the class of bug that has now bitten twice in this program: a variable
computed in main() and referenced inside a helper without being passed in.
Python only raises at runtime, and only on the code path that touches it.
"""
import ast, builtins, sys

def check(path):
    tree = ast.parse(open(path).read(), path)
    module_names = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}

    # Collect MODULE-level bindings only.  The previous version used
    # ast.walk(tree), which descends into function bodies, so a variable
    # assigned only inside main() counted as a global -- which made this
    # checker unable to catch the very bug class it was written for.  Descend
    # through top-level if/try/for/with, but never into a function or class
    # body.
    SKIP = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    def _collect(node) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                for a in child.names:
                    module_names.add((a.asname or a.name).split(".")[0])
            elif isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (child.targets if isinstance(child, ast.Assign)
                           else [child.target])
                for t in targets:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            module_names.add(x.id)
            elif isinstance(child, SKIP):
                module_names.add(child.name)
                continue  # do not descend into its body
            _collect(child)

    _collect(tree)

    # A nested function sees the enclosing function's parameters and locals.
    # Without this, every closure over an enclosing parameter looked undefined
    # (e.g. aggregate.py's `cell()` closing over `paired_contrast`'s `metric`).
    FN = (ast.FunctionDef, ast.AsyncFunctionDef)
    enclosing: dict[ast.AST, list[ast.AST]] = {}
    def _descend(node, stack):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, FN):
                enclosing[child] = list(stack)
                _descend(child, stack + [child])
            else:
                _descend(child, stack)
    _descend(tree, [])

    def _bound_by(fn) -> set[str]:
        names = set()
        for a in (list(fn.args.args) + list(fn.args.kwonlyargs)
                  + list(fn.args.posonlyargs)):
            names.add(a.arg)
        for a in (fn.args.vararg, fn.args.kwarg):
            if a:
                names.add(a.arg)
        for n in ast.walk(fn):
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            names.add(x.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                # A function-local import binds a name for its closures too
                # (measure_curvature.py imports mujoco inside the function whose
                # nested body_name_of uses it).
                for a in n.names:
                    names.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
        return names

    bad = []
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        local = set(module_names)
        for outer in enclosing.get(fn, []):
            local |= _bound_by(outer)
        for a in list(fn.args.args) + list(fn.args.kwonlyargs) + list(fn.args.posonlyargs):
            local.add(a.arg)
        for a in (fn.args.vararg, fn.args.kwarg):
            if a: local.add(a.arg)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store,)):
                local.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names: local.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, ast.ExceptHandler) and n.name:
                local.add(n.name)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local.add(n.name)
                # nested function parameters are in scope inside that closure
                if not isinstance(n, ast.ClassDef):
                    for a in (list(n.args.args) + list(n.args.kwonlyargs)
                              + list(n.args.posonlyargs)):
                        local.add(a.arg)
                    for a in (n.args.vararg, n.args.kwarg):
                        if a: local.add(a.arg)
            elif isinstance(n, (ast.comprehension,)):
                for x in ast.walk(n.target):
                    if isinstance(x, ast.Name): local.add(x.id)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in local:
                bad.append(f"{path}:{n.lineno}: {fn.name}() uses undefined '{n.id}'")
    return bad

issues = []
for f in sys.argv[1:]:
    issues += check(f)
print("\n".join(sorted(set(issues))) if issues else "no undefined names in function scope")
sys.exit(1 if issues else 0)
