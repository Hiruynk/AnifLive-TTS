"""Extract specific pinned official definitions for frontend and reference DSP."""
import __future__
import ast


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    chosen = [node for node in tree.body
              if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in chosen} != set(names):
        raise ValueError(f'Missing official definitions: {names}')
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
