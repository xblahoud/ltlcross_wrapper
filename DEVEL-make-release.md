# How to make a release

## 1. Update version
Update new version in:
 * [init file](ltlcross_runner/__init__.py)
 * [setup.py](setup.py)
 * [Changelog](CHANGELOG.md)
 
## Synchronize with git
 * merge next to master
 * push master
 * make tag on gitlab (and copy Changelog entries to release notes)
 * make sure the links in Changelog work with the tag
 
## Build & upload to PyPI
```
rm -rf dist build ltlcross_runner.egg-info
python3 setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*
```
