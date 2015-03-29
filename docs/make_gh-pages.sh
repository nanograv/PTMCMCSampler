#!/bin/bash

set -e

ID=`git log master -1 --pretty=short --abbrev-commit`

GH_PAGES_SOURCES='docs PTMCMCSampler'

cd ../
git checkout gh-pages
rm -rf build _sources _static
git checkout master $(GH_PAGES_SOURCES)
git reset HEAD
cd docs
make html
cd ../
mv -fv docs/_build/html/* .
rm -rf $(GH_PAGES_SOURCES)
git add -A
git commit -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`"
git push origin gh-pages
git checkout master
