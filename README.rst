Machine learning course notes
=============================

Convert the notebook to slides with::

  jupyter nbconvert Machine\ Learning.ipynb --to slides --post serve --SlidesExporter.reveal_scroll=True --SlidesExporter.reveal_theme=solarized

or, using a file-system-watcher like ``entr`` with::

  ls Machine\ Learning.ipynb | entr venv/bin/jupyter nbconvert Machine\ Learning.ipynb --to slides --SlidesExporter.reveal_scroll=True --SlidesExporter.reveal_theme=solarized

and browse to the file manually in your browser (after downlading and extracting reveal.js to this directory)

