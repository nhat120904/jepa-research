"""Minimal undefined-name checker for function scopes.

Catches the class of bug that has now bitten twice in this program: a variable
computed in main() and referenced inside a helper without being passed in.
Python only raises at runtime, and only on the code path that touches it.
"""
import ast, builtins, sys

def check(path):
    tree = ast.parse(open(path).read(), path)
    module_names = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                module_names.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        module_names.add(x.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_names.add(n.name)
    bad = []
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        local = set(module_names)
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
