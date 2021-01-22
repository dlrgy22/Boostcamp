import pathlib

cwd = pathlib.Path.cwd()
print(cwd)
parent = cwd.parent
print(parent)
parents = list(cwd.parents)
print(parents)
