.. _new-object:

Adding an Object
================

In mjlab, everything is an ``Entity`` – whether it is a robot, a table, a shelf,
or any other object. This unified abstraction makes it straightforward to add
static objects to your environments using the same API you use for robots.

Below is an example showing how to add a static box (e.g., a table) to the
tracking task.

Example: adding a static box
----------------------------

.. code-block:: python

   import mujoco

   from mjlab.asset_zoo.robots import get_g1_robot_cfg
   from mjlab.entity import EntityCfg
   from mjlab.tasks.tracking import make_tracking_env_cfg

   def create_box() -> mujoco.MjSpec:
       spec = mujoco.MjSpec()
       body = spec.worldbody.add_body(name="box")
       body.add_geom(
           type=mujoco.mjtGeom.mjGEOM_BOX,
           size=(0.5, 0.5, 0.05),
           rgba=(0.8, 0.3, 0.3, 1.0),
       )
       return spec

   def get_box_cfg() -> EntityCfg:
       return EntityCfg(
           init_state=EntityCfg.InitialStateCfg(
               pos=(0.0, 0.0, 0.3),
               rot=(1.0, 0.0, 0.0, 0.0),
           ),
           spec_fn=create_box,
       )

   cfg = make_tracking_env_cfg()
   cfg.scene.entities = {
       "robot": get_g1_robot_cfg(),
       "table": get_box_cfg(),
   }

Notes:

- For static objects, ``init_state.pos`` and ``init_state.rot`` define the
  **fixed pose** in the world.
- As long as the object has no actuators and no non-zero joint DOFs, it will
  behave as a fixed obstacle/table.

Randomizing object pose (mocap bodies)
--------------------------------------

If you want to randomize the object pose at reset time, set ``mocap=True`` on
the root body in your ``MjSpec``. This allows mjlab to change its pose at
runtime via events:

.. code-block:: python

   def create_box() -> mujoco.MjSpec:
       spec = mujoco.MjSpec()
       body = spec.worldbody.add_body(name="box")
       body.mocap = True  # Enable mocap for runtime pose changes.
       body.add_geom(
           type=mujoco.mjtGeom.mjGEOM_BOX,
           size=(0.5, 0.5, 0.05),
           rgba=(0.8, 0.3, 0.3, 1.0),
       )
       return spec

With ``mocap=True`` on the root body, you can use standard event functions
(such as a ``reset_root_state_uniform``-style event) to randomize the object's
pose across environments at reset:

- Sample random positions / orientations,
- Apply them to the mocap body at the beginning of each episode,
- Keep the rest of your task configuration unchanged.

EntityCfg works for all object types
------------------------------------

The same ``EntityCfg`` API works for a wide range of entities:

- Fixed non-articulated objects (tables, walls, shelves),
- Fixed articulated systems (robot arms bolted to the world, doors, drawers),
- Floating non-articulated objects (boxes, mugs, balls),
- Floating articulated systems (humanoids, quadrupeds).

You can add as many entities as you want to ``cfg.scene.entities``; mjlab will
handle parallelization across environments automatically.
