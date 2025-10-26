import os
import sys

import sphinx_book_theme

sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("../src/mjlab"))

# -- Project information -----------------------------------------------------

project = "mjlab"
copyright = "2025, The mjlab Developers"
author = "The mjlab Developers"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  "autodocsumm",
  "myst_parser",
  "sphinx.ext.napoleon",
  "sphinxemoji.sphinxemoji",
  "sphinx.ext.autodoc",
  "sphinx.ext.autosummary",
  "sphinx.ext.githubpages",
  "sphinx.ext.intersphinx",
  "sphinx.ext.mathjax",
  "sphinx.ext.todo",
  "sphinx.ext.viewcode",
  "sphinxcontrib.bibtex",
  "sphinxcontrib.icon",
  "sphinx_copybutton",
  "sphinx_design",
  "sphinx_tabs.tabs",  # backwards compatibility for building docs on v1.0.0
  "sphinx_multiversion",
]


# mathjax hacks
mathjax3_config = {
  "tex": {
    "inlineMath": [["\\(", "\\)"]],
    "displayMath": [["\\[", "\\]"]],
  },
}

# panels hacks
panels_add_bootstrap_css = False
panels_add_fontawesome_css = True

# supported file extensions for source files
source_suffix = {
  ".rst": "restructuredtext",
  ".md": "markdown",
}

# make sure we don't have any unknown references
# TODO: Enable this by default once we have fixed all the warnings
nitpick_ignore = [
  ("py:obj", "slice(None)"),
]

nitpick_ignore_regex = [
  (r"py:.*", r"pxr.*"),  # we don't have intersphinx mapping for pxr
  (r"py:.*", r"trimesh.*"),  # we don't have intersphinx mapping for trimesh
]

# emoji style
sphinxemoji_style = "twemoji"  # options: "twemoji" or "unicode"
# put type hints inside the signature instead of the description (easier to maintain)
autodoc_typehints = "signature"  # "description"
# autodoc_typehints_format = "fully-qualified"
# document class *and* __init__ methods
autoclass_content = "class"  #
# separate class docstring from __init__ docstring
autodoc_class_signature = "separated"
# sort members by source order
autodoc_member_order = "bysource"
# inherit docstrings from base classes
autodoc_inherit_docstrings = True
# BibTeX configuration
bibtex_bibfiles = ["source/_static/refs.bib"]
# generate autosummary even if no references
autosummary_generate = True
autosummary_generate_overwrite = False
# default autodoc settings
autodoc_default_options = {
  "members": True,
  # "undoc-members": True,
  # "inherited-members": True,
  # "show-inheritance": True,
}

# generate links to the documentation of objects in external projects
intersphinx_mapping = {
  "python": ("https://docs.python.org/3", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = []
# autodoc_type_aliases = {
#     'Tensor': 'torch.Tensor',
# }
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
  "_build",
  "_redirect",
  "_templates",
  "Thumbs.db",
  ".DS_Store",
  "README.md",
  "licenses/*",
]

# Mock out modules that are not available on RTD
autodoc_mock_imports = [
  "torch",
  "numpy",
  "matplotlib",
  "scipy",
  "carb",
  "warp",
  "pxr",
  "h5py",
  "hid",
  "prettytable",
  "tqdm",
  "tensordict",
  "trimesh",
  "toml",
  "mujoco_warp",
  "gymnasium",
  "rsl_rl",
  "viser",
  "wandb",
]

# List of zero or more Sphinx-specific warning categories to be squelched (i.e.,
# suppressed, ignored).
suppress_warnings = [
  # Generally speaking, we do want Sphinx to inform
  # us about cross-referencing failures. Remove this entirely after Sphinx
  # resolves this open issue:
  #   https://github.com/sphinx-doc/sphinx/issues/4961
  # Squelch mostly ignorable warnings resembling:
  #     WARNING: more than one target found for cross-reference 'TypeHint':
  #     beartype.door._doorcls.TypeHint, beartype.door.TypeHint
  #
  # Sphinx currently emits *MANY* of these warnings against our
  # documentation. All of these warnings appear to be ignorable. Although we
  # could explicitly squelch *SOME* of these warnings by canonicalizing
  # relative to absolute references in docstrings, Sphinx emits still others
  # of these warnings when parsing PEP-compliant type hints via static
  # analysis. Since those hints are actual hints that *CANNOT* by definition
  # by canonicalized, our only recourse is to squelch warnings altogether.
  "ref.python",
  # 'autosummary'
]

# -- Internationalization ----------------------------------------------------

# specifying the natural language populates some key tags
language = "en"

# -- Options for HTML output -------------------------------------------------


html_title = "mjlab Documentation"
html_theme_path = [sphinx_book_theme.get_html_theme_path()]
html_theme = "sphinx_book_theme"
# html_favicon = "source/_static/favicon.ico"
html_show_copyright = True
html_show_sphinx = False
html_last_updated_fmt = ""  # to reveal the build date in the pages meta

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["source/_static/css"]
html_css_files = ["custom.css"]

html_theme_options = {
  "path_to_docs": "docs_web/",
  "collapse_navigation": True,
  "repository_url": "https://github.com/BDX-R/BDX-R.github.io",
  "use_repository_button": True,
  "use_issues_button": True,
  "use_edit_page_button": True,
  "show_toc_level": 1,
  "use_sidenotes": True,
  "logo": {
    "text": "The mjlab Documentation",
    # "image_light": "source/_static/mjlab-banner.jpg",
    # "image_dark": "source/_static/mjlab-banner.jpg",
  },
  "icon_links": [
    {
      "name": "GitHub",
      "url": "https://github.com/mujocolab/mjlab",
      "icon": "fa-brands fa-square-github",
      "type": "fontawesome",
    },
    {
      "name": "Stars",
      "url": "https://img.shields.io/github/stars/mujocolab/mjlab?color=fedcba",
      "icon": "https://img.shields.io/github/stars/mujocolab/mjlab?color=fedcba",
      "type": "url",
    },
  ],
  "icon_links_label": "Quick Links",
}

templates_path = [
  "_templates",
]

# Whitelist pattern for remotes
smv_remote_whitelist = r"^.*$"
# Whitelist pattern for branches (set to None to ignore all branches)
smv_branch_whitelist = os.getenv("SMV_BRANCH_WHITELIST", r"^(main|devel)$")
# Whitelist pattern for tags (set to None to ignore all tags)Add commentMore actions
smv_tag_whitelist = os.getenv("SMV_TAG_WHITELIST", r"^v[1-9]\d*\.\d+\.\d+$")
# html_sidebars = {
#     "**": ["navbar-logo.html", "versioning.html", "icon-links.html", "search-field.html", "sbt-sidebar-nav.html"]
# }

html_sidebars = {
  "**": [
    "navbar-logo.html",
    "icon-links.html",
    "search-field.html",
    "sbt-sidebar-nav.html",
  ]
}


# -- Advanced configuration -------------------------------------------------


def skip_member(app, what, name, obj, skip, options):
  # List the names of the functions you want to skip here
  exclusions = ["from_dict", "to_dict", "replace", "copy", "validate", "__post_init__"]
  if name in exclusions:
    return True
  return None


def setup(app):
  app.connect("autodoc-skip-member", skip_member)
