.. _installation:

Installation Guide
==================

mjlab is in active **beta** and tightly coupled to MuJoCo Warp.  
This guide presents one **recommended installation path** and then lists alternatives for other setups.

System Requirements
*******************

- **Python**: 3.10 or higher
- **Operating System**:
    - **Linux** – recommended
    - **macOS** – supported for **evaluation only**
    - **Windows** – supported but **experimental**
- **GPU**: NVIDIA GPU **strongly recommended** 
- **CUDA version**: CUDA 12.4+ recommended

Prerequisite: ``uv``
********************

mjlab is developed and tested primarily with `uv`_, a fast Python package and project manager.

If you do not have ``uv`` installed, run:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh


Choosing an Installation Method
*******************************

Most users should use **Method 1 – From source (with uv)**.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - You are…
     - Recommended method
   * - Trying mjlab and/or contributing to mjlab itself
     - **Method 1** – From source (with ``uv``)
   * - Using mjlab as a dependency in your own project managed by ``uv``
     - **Method 2** – From PyPI / Git (with ``uv``)
   * - Using classic ``pip`` / ``venv`` / ``conda`` (no ``uv``)
     - **Method 3** – pip-based installation
   * - Running in containers / on clusters
     - **Method 4** – Docker

Method 1 – From Source (Recommended)
------------------------------------

Use this if you want the **latest** beta and/or to contribute to mjlab.

**Clone the repository**

.. code-block:: bash

   git clone https://github.com/mujocolab/mjlab.git
   cd mjlab

**Create and activate a virtual environment with ``uv``**

.. code-block:: bash

   uv venv .venv
   source .venv/bin/activate

(If you already have a project-level ``.venv``, you can reuse it instead.)

**Install mjlab (editable) with dependencies**

Inside the cloned repository:

.. code-block:: bash

   uv add --editable .

This will:

- Install ``mjlab`` in editable mode into the active environment,
- Install its dependencies, including a pinned ``mujoco-warp`` revision.

**Sanity check**

Still inside the ``mjlab`` repository:

.. code-block:: bash

   uv run demo

This should launch the demo (viewer and/or logging).  
If this works, your installation is functional and you are ready to follow the tutorials.

Method 2 – Using ``uv`` in Your Own Project
-------------------------------------------

Use this if mjlab is a **dependency** of a separate project that you manage with ``uv``.

**From Git (latest HEAD – recommended during beta)**

Inside your project directory (with an existing ``uv``-managed environment):

.. code-block:: bash

   uv add \
     "mjlab @ git+https://github.com/mujocolab/mjlab" \
     "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@fb9bf88399796f161a4a2b129d846484da8a4ad0"

Notes:

- ``mjlab`` is installed directly from the GitHub repository (latest ``main`` or default branch).
- ``mujoco-warp`` is installed from Git at a known-good commit.
- This is the best option if you want to keep up with mjlab development.

**From PyPI (beta snapshot)**

If you prefer to depend on the published beta snapshot of mjlab on PyPI:

.. code-block:: bash

   uv add \
     mjlab \
     "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@486642c3fa262a989b482e0e506716d5793d61a9"

Caveats:

- The PyPI package is a **snapshot**, not a stable release.
- You still **must** install ``mujoco-warp`` from Git, because it is not on PyPI.
- The pinned commit for ``mujoco-warp`` should match the snapshot version you are targeting.

Method 3 – pip-based Installation (venv, conda, virtualenv, …)
--------------------------------------------------------------

If you cannot or do not want to use ``uv``, you can install mjlab with ``pip``.

Because ``mujoco-warp`` is only available via Git, there is an important constraint:

.. warning::

   ``pip`` cannot resolve the Git dependency for ``mujoco-warp`` as declared in ``pyproject.toml``.  
   You **must install ``mujoco-warp`` from Git manually** before installing mjlab.

**Create and activate a virtual environment**

Using ``venv`` (standard library):

.. code-block:: bash

   python -m venv mjlab-env
   source mjlab-env/bin/activate

Using conda:

.. code-block:: bash

   conda create -n mjlab python=3.11
   conda activate mjlab

(Any supported Python version ≥ 3.10 is acceptable; 3.11 is a good default.)

**Install ``mujoco-warp`` and mjlab**

From source (recommended)

.. code-block:: bash

   # Install mujoco-warp from Git (known-good commit)
   pip install "git+https://github.com/google-deepmind/mujoco_warp@fb9bf88399796f161a4a2b129d846484da8a4ad0"

   # Clone mjlab and install it in editable mode
   git clone https://github.com/mujocolab/mjlab.git
   cd mjlab
   pip install -e .

From PyPI (beta snapshot)

.. code-block:: bash

   pip install "git+https://github.com/google-deepmind/mujoco_warp@486642c3fa262a989b482e0e506716d5793d61a9"
   pip install mjlab

Method 4 – Docker
-----------------

A Dockerfile and helper script are provided for containerized use.

**Prerequisites**

1. Install Docker:

   - https://docs.docker.com/engine/install/

2. Install an appropriate NVIDIA driver and the NVIDIA Container Toolkit:

   - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

   Make sure you:

   - Register the NVIDIA runtime with Docker,
   - Restart Docker as described in the NVIDIA documentation.

**Build the image**

From the cloned mjlab repository:

.. code-block:: bash

   make docker-build

**Run the container**

Use the helper script to run an ``mjlab`` container:

.. code-block:: bash

   ./scripts/run_docker.sh

Examples:

- Demo with viewer:

  .. code-block:: bash

     ./scripts/run_docker.sh uv run demo

- Training example:

  .. code-block:: bash

     ./scripts/run_docker.sh \
       uv run train Mjlab-Velocity-Flat-Unitree-G1 \
       --env.scene.num-envs 4096

Verification
------------

After installing mjlab by **any** method, verify that it works by running the demo.

**With ``uv``**

If you are inside the mjlab repository or a project that depends on mjlab:

.. code-block:: bash

   uv run demo

or explicitly:

.. code-block:: bash

   uv run python -m mjlab.scripts.demo

**With plain ``pip`` / conda**

If mjlab is installed in your active environment:

.. code-block:: bash

   # If a console script entry point is defined:
   demo

   # Or using the module form (always available):
   python -m mjlab.scripts.demo

If the demo runs without errors, your installation is working correctly.

Troubleshooting
---------------

make it shorter
1. **Check the FAQ**

   Consult the mjlab FAQ for answers to common installation and runtime issues
   (path depends on your docs structure, e.g. :doc:`FAQ <../project/faq>`).

2. **CUDA / GPU issues**

   - Verify that your NVIDIA driver is installed and up to date.
   - Confirm that your CUDA version is compatible with the ``mujoco-warp`` commit
     you are using (see:
     https://github.com/google-deepmind/mujoco_warp/issues/101).

3. **macOS slowness**

   - Training is not supported on macOS (no GPU acceleration).
   - Evaluation may be significantly slower compared to Linux with an NVIDIA GPU.

4. **Still stuck?**

   - Open an issue on GitHub:
     https://github.com/mujocolab/mjlab/issues

   Please include:

   - Operating system and version,
   - Python version,
   - GPU model and driver version,
   - Installation method (source / PyPI / pip / Docker),
   - The exact command you ran and the full error message / stack trace.
