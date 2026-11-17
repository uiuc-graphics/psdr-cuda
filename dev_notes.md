# Dev Notes

## Publish to PyPI
The following was taken from [this guide](https://realpython.com/pypi-publish-python-package/#publish-your-package-to-pypi).

Install the necessary packages:
```bash
pip install build twine
```
Create a source archive and a wheel for your package:
```bash
python -m build
```
Check with `twine`:
```bash
twine check dist/*
```

If you would like to do a test upload:
```bash
twine upload -r testpypi dist/*
```

Final upload:
```bash
twine upload dist/*
```
