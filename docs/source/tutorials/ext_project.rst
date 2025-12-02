.. _ext-project:

Creating an External Project
============================

In this tutorial, you will create a minimal **external mjlab project**.  
The goal is to develop your own tasks in a separate repository, while still
using all the mjlab tooling (CLI, configs, etc.).

By the end, you will have:

- a standalone Python package managed by ``uv``
- ``mjlab`` and ``mujoco-warp`` installed as dependencies
- a ``tasks`` package automatically discovered by mjlab

Step 1 – Install ``uv``
^^^^^^^^^^^^^^^^^^^^^^^

If you do not have ``uv`` installed, install it with:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Step 2 – Create the project
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a new Python package and move into it:

.. code-block:: bash

   uv init --package mjlab_cookbook_project
   cd mjlab_cookbook_project

This creates a minimal Python package under ``src/mjlab_cookbook_project/`` and
a ``pyproject.toml`` managed by ``uv``.

Step 3 – Add mjlab as a dependency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add ``mjlab`` and ``mujoco-warp`` as dependencies of your project.  
Here we install them directly from their Git repositories (see :ref:`installation`
for alternative methods):

.. code-block:: bash

   uv add "mjlab @ git+https://github.com/mujocolab/mjlab" \
          "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@486642c3fa262a989b482e0e506716d5793d61a9"

Step 4 – Define a global project path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We define a global constant pointing to the root of your external project’s
source tree. This is convenient for locating assets, configs, etc.

Edit ``src/mjlab_cookbook_project/__init__.py`` and add:

.. code-block:: python

   from pathlib import Path

   MJLAB_COOKBOOK_PROJECT_SRC_PATH: Path = Path(__file__).parent

Step 5 – Initialize the tasks package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create the directory that will hold your custom tasks:

.. code-block:: bash

   mkdir src/mjlab_cookbook_project/tasks

Then create the file ``src/mjlab_cookbook_project/tasks/__init__.py`` with:

.. code-block:: python

   from mjlab.third_party.isaaclab.isaaclab_tasks.utils.importer import import_packages

   _BLACKLIST_PKGS = ["utils", ".mdp"]

   import_packages(__name__, _BLACKLIST_PKGS)

This makes ``mjlab_cookbook_project.tasks`` behave like the built-in mjlab tasks
package: all subpackages (except those in the blacklist) will be discovered and
imported automatically.

Step 6 – Register your tasks as an mjlab plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, expose your tasks package through an **entry point** so that mjlab can
discover it when you run its commands (e.g. ``mjlab run ...``).

Add the following section to your ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."mjlab.tasks"]
   mjlab_cookbook_project = "mjlab_cookbook_project.tasks"

With this in place, mjlab will load tasks from your external project as if they
were defined inside the main mjlab repository.

Next steps
^^^^^^^^^^

Once that is done, we recommend continuing with :ref:`new-robot` to add a custom
robot and define your first task.
