# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'PyRID'
copyright = '2022, Moritz F P Becker'
author = 'Moritz F P Becker'

# The full version, including alpha/beta/rc tags
release = '15.06.2022'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'numpydoc', 'sphinxcontrib.bibtex']#, 'sphinx.ext.imgmath']

#imgmath_image_format = 'svg'

#extensions = [
#    'sphinx.ext.viewcode',
#	 'sphinx.ext.todo',
#	#'sphinx.ext.imgmath',
#	'sphinxcontrib.bibtex',
#	'sphinxcontrib.plantuml',
#	'sphinxcontrib.exceltable',
#    'sphinx.ext.autodoc',
#    'numpydoc',
#    'sphinx.ext.intersphinx',
#    'sphinx.ext.coverage',
#    'sphinx.ext.doctest',
#    'sphinx.ext.autosummary',
#    'sphinx.ext.graphviz',
#    'sphinx.ext.ifconfig',
#    'matplotlib.sphinxext.plot_directive',
#    'IPython.sphinxext.ipython_console_highlighting',
#    'IPython.sphinxext.ipython_directive',
#    'sphinx.ext.mathjax',
#    'sphinx_panels',
#]

numfig = True

autodoc_default_flags = ['members', 'inherited-members']

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# autodoc_mock_imports = ["numba"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'#'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'Wiser.tex', u'Wiser Documentation',
   u'the Nile team', 'manual'),
]

bibtex_bibfiles = ['Library/PyRID.bib']
bibtex_default_style = 'unsrt' # 'alpha' # 'plain' # 'unsrtalpha' # 

html_logo = "_static/PyRID_Logo_Render2_cropped.png"

def setup(app):
    app.add_css_file('my_theme.css')

