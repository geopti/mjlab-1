.. _new-robot:

Adding a Robot
==============

In this tutorial, you will add a simple **CartPole** robot to your external
project. You will:

- create a MuJoCo XML describing the robot
- define an ``EntityCfg`` so mjlab can build and control it
- register the robot so mjlab can discover and reuse it

Prerequisite
^^^^^^^^^^^^

We assume you already have an external project set up as in :ref:`ext-project`.

Step 1 - Go to your project
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd path/to/mjlab_cookbook_project

Step 2 - Create the robot folder structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You will now create a ``robots`` package and a dedicated subpackage for the
CartPole robot.

In ``src/mjlab_cookbook_project/``, create the following structure:

.. code-block:: bash

   src/mjlab_cookbook_project/
     __init__.py
     robots/
       __init__.py
       cartpole/
         __init__.py
         cartpole_constants.py
         xmls/
           cartpole.xml

Concretely, you can do:

.. code-block:: bash

   mkdir -p src/mjlab_cookbook_project/robots/cartpole/xmls

   touch src/mjlab_cookbook_project/robots/__init__.py
   touch src/mjlab_cookbook_project/robots/cartpole/__init__.py

You will fill in ``cartpole.xml`` and ``cartpole_constants.py`` in the next
steps.

Step 3 - Write the CartPole MuJoCo XML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The XML defines the physical model: a cart sliding along the x-axis, with a
hinged pole attached on top.

Create ``src/mjlab_cookbook_project/robots/cartpole/xmls/cartpole.xml`` with:

.. code-block:: xml

   <mujoco model="cartpole">
     <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
     <worldbody>
       <geom name="ground" type="plane" pos="0 0 0" size="5 5 0.1" rgba="0.8 0.9 0.8 1"/>
       <body name="cart" pos="0 0 0.1">
         <geom type="box" size="0.2 0.1 0.1" rgba="0.2 0.2 0.8 1" mass="1.0"/>
         <joint name="slide" type="slide" axis="1 0 0" limited="true" range="-2 2"/>
         <body name="pole" pos="0 0 0.1">
           <geom type="capsule" size="0.05 0.5" fromto="0 0 0 0 0 1" rgba="0.8 0.2 0.2 1" mass="2.0"/>
           <joint name="hinge" type="hinge" axis="0 1 0" range="-90 90"/>
         </body>
       </body>
     </worldbody>
     <actuator>
       <velocity name="slide_velocity" joint="slide" ctrlrange="-20 20" kv="20"/>
     </actuator>
     <keyframe>
       <key name="cartpole_init" qpos="0 0" qvel="0 0" ctrl="0 0"/>
     </keyframe>
   </mujoco>

Step 4 - Define the CartPole configuration in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now you will wrap the XML into an mjlab ``EntityCfg``. This tells mjlab how to:

- load the MuJoCo spec
- attach actuators
- build an ``Entity`` for simulation

Create
``src/mjlab_cookbook_project/robots/cartpole/cartpole_constants.py`` with:

.. code-block:: python

   from pathlib import Path
   import mujoco

   from mjlab_cookbook_project import MJLAB_COOKBOOK_PROJECT_SRC_PATH
   from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
   from mjlab.actuator import BuiltinPositionActuatorCfg

   # Path to the CartPole XML
   CARTPOLE_XML: Path = (
       MJLAB_COOKBOOK_PROJECT_SRC_PATH
       / "robots"
       / "cartpole"
       / "xmls"
       / "cartpole.xml"
   )
   assert CARTPOLE_XML.exists(), f"XML not found: {CARTPOLE_XML}"

   # MuJoCo spec loader
   def get_spec() -> mujoco.MjSpec:
       return mujoco.MjSpec.from_file(str(CARTPOLE_XML))

   # Actuator configuration for the sliding joint
   CARTPOLE_ACTUATOR = BuiltinPositionActuatorCfg(
       joint_names_expr=["slide"],
       effort_limit=20.0,
       stiffness=0.0,
       damping=0.1,
   )

   # Articulation (how actuators are attached)
   CARTPOLE_ARTICULATION = EntityArticulationInfoCfg(
       actuators=(CARTPOLE_ACTUATOR,),
   )

   # Public robot configuration used by mjlab
   CARTPOLE_ROBOT_CFG = EntityCfg(
       spec_fn=get_spec,
       articulation=CARTPOLE_ARTICULATION,
   )

   # Quick local test: launch the MuJoCo viewer
   if __name__ == "__main__":
       import mujoco.viewer as viewer

       robot = Entity(CARTPOLE_ROBOT_CFG)
       viewer.launch(robot.spec.compile())

Step 5 - Register the robot in the robots package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Expose the CartPole configuration from the top-level ``robots`` package so that
other parts of your project (and mjlab tasks) can import it in a consistent way.

Edit ``src/mjlab_cookbook_project/robots/__init__.py``:

.. code-block:: python

   from mjlab_cookbook_project.robots.cartpole.cartpole_constants import CARTPOLE_ROBOT_CFG

   __all__ = (
       "CARTPOLE_ROBOT_CFG",
   )

This allows you to import the robot config as:

.. code-block:: python

   from mjlab_cookbook_project.robots import CARTPOLE_ROBOT_CFG

Step 6 - Verify the setup
^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, verify that everything loads correctly. The test script inside
``cartpole_constants.py``:

- builds an ``Entity`` from ``CARTPOLE_ROBOT_CFG``
- compiles the MuJoCo spec into a model
- launches the MuJoCo viewer

Run:

.. code-block:: bash

   uv run python src/mjlab_cookbook_project/robots/cartpole/cartpole_constants.py

You should see the CartPole model in the MuJoCo viewer.

.. image:: ../_static/content/cartpole-env.jpg
   :width: 100%
   :alt: CartPole environment

Next, you can move on to defining a task that uses ``CARTPOLE_ROBOT_CFG`` and
hook it into the mjlab task system.
