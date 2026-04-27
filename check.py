import ast

files = ['src/model.py', 'src/train.py', 'src/evaluate.py']

for f in files:
    tree = ast.parse(open(f).read())
    names = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    print(f"{f}: {names}")