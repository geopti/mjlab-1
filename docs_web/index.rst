Welcome to mjlab!
=================

.. figure:: source/_static/mjlab-banner.jpg
   :width: 100%
   :alt: mjlab

What is mjlab?
==============

**mjlab = Isaac Lab's API + MuJoCo's simplicity + GPU acceleration**

mjlab takes Isaac Lab's proven manager-based architecture and RL abstractions,  
and builds them directly on top of MuJoCo Warp. No translation layer, no Omniverse  
runtime, just fast and transparent physics.

- **Familiar API** for Isaac Lab users (managers, configs, tasks).
- **GPU-accelerated MuJoCo** via MuJoCo Warp for large-scale RL.
- **Open-source and research-friendly**, with a clean Python API.

Who is mjlab for?
=================

mjlab is designed for:

- **RL researchers** who want Isaac Lab–style tasks and abstractions without Omniverse.
- **Robotics engineers** who already use MuJoCo and want a scalable RL stack on top.
- **Isaac Lab / Isaac Gym users** looking for a lighter-weight, more transparent backend.

System requirements
===================

.. note::

   - **GPU**: NVIDIA GPU with recent drivers (for MuJoCo Warp acceleration).
   - **OS**: Linux recommended. macOS is supported for evaluation only (significantly slower).
   - **Python**: 3.11 (other versions may work but are not officially supported yet).

Quick start
===========

You can try mjlab *without installing anything* by using `uvx`:

.. code-block:: bash

   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Run the mjlab demo (no local installation needed)
   uvx --from mjlab \
       --with "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@486642c3fa262a989b482e0e506716d5793d61a9" \
       demo

If this runs and opens a viewer / logs a rollout, your setup is compatible with mjlab.

Next steps
==========

- :doc:`Getting started / Installation guide <source/getting_started/installation>`
- :doc:`Tutorial: Create a task <source/tutorials/create_new_task>`
- :doc:`Migration from Isaac Lab <source/getting_started/migration_isaac_lab>`
- :doc:`Migration from Isaac Gym <source/getting_started/migration_isaac_gym>`

License & citation
==================

mjlab is licensed under the Apache License, Version 2.0.  
Please refer to the `LICENSE file <https://github.com/mujocolab/mjlab/blob/main/LICENSE/>`_ for details.

If you use mjlab in your research, we would appreciate a citation:

.. code-block:: bibtex

    @software{Zakka_MJLab_Isaac_Lab_2025,
        author = {Zakka, Kevin and Yi, Brent and Liao, Qiayuan and Le Lay, Louis},
        license = {Apache-2.0},
        month = sep,
        title = {{MJLab: Isaac Lab API, powered by MuJoCo-Warp, for RL and robotics research.}},
        url = {https://github.com/mujocolab/mjlab},
        version = {0.1.0},
        year = {2025}
    }

Acknowledgments
===============

mjlab would not exist without the excellent work of the Isaac Lab team, whose API design
and abstractions mjlab builds upon.

Thanks also to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for 
answering our questions, giving helpful feedback, and implementing features based 
on our requests countless times.

Table of Contents
=================

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   source/getting_started/installation
   source/getting_started/migration_isaac_lab
   source/getting_started/migration_isaac_gym

.. toctree::
   :maxdepth: 1
   :caption: About the Project

   source/project/motivation
   source/project/faq

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   source/tutorials/create_new_task

.. toctree::
   :maxdepth: 1
   :caption: Core Concepts

   source/core/domain_randomization
   source/core/nan_guard
   source/core/obs_history_delay
   source/core/spec_config

.. toctree::
   :maxdepth: 1
   :caption: Source API

   source/api/index