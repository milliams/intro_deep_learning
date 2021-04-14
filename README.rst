Machine learning course notes
=============================

Run the examples for yourself
-----------------------------

The easiest way to run the example code on your local computer is to create a virtual environment and run the notebook inside it:

.. code-block:: shell-session

   python3 -m venv dl_venv
   dl_venv/bin/pip install -r requirements.txt

Start a JupyterLab session with the Iris notebook opened:

.. code-block:: shell-session

   dl_venv/bin/jupyter-lab iris.ipynb

You can do the same with ``mnist.ipynb`` or open it from within JupyterLab's file browser.

If you want to make use of your local GPU then you will need to make sure that you have installed CUDA.

Generate the slides
-------------------

Convert the notebook to slides with::

  jupyter nbconvert intro_deep_learning.ipynb --to slides --post serve --SlidesExporter.reveal_scroll=True --SlidesExporter.reveal_theme=solarized

or, using a file-system-watcher like ``entr`` with::

  ls intro_deep_learning.ipynb | entr venv/bin/jupyter nbconvert intro_deep_learning.ipynb --to slides --SlidesExporter.reveal_scroll=True --SlidesExporter.reveal_theme=solarized

and browse to the file manually in your browser. 
