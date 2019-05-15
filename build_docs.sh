#!/usr/bin/env bash
rm -rf docs
pdoc3 --output-dir docs --html quantfin
mv docs/quantfin/* docs/
rm -rf docs/quantfin

# pdoc outputs files with <!doctype.html>, whereas gh-pages requires
# <!DOCTYPE html> on the index page. This fixes that
printf '%s\n%s\n' "<!DOCTYPE html>" "$(tail -n +2 docs/index.html)" > docs/index.html
