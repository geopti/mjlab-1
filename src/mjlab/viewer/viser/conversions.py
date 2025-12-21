"""Convert MuJoCo mesh data to trimesh format with texture support."""

import mujoco
import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj
from PIL import Image

# Default colors for geoms without materials.
_DEFAULT_COLLISION_COLOR = np.array([204, 102, 102, 128], dtype=np.uint8)
_DEFAULT_VISUAL_COLOR = np.array([31, 128, 230, 255], dtype=np.uint8)


def rgba_to_uint8(rgba: np.ndarray) -> np.ndarray:
  """Convert RGBA from [0,1] range to [0,255] uint8."""
  return (rgba * 255).astype(np.uint8)


def mujoco_mesh_to_trimesh(
  mj_model: mujoco.MjModel, geom_idx: int, verbose: bool = False
) -> trimesh.Trimesh:
  """Convert a MuJoCo mesh geometry to a trimesh with textures if available.

  Args:
    mj_model: MuJoCo model object
    geom_idx: Index of the geometry in the model
    verbose: If True, print debug information during conversion

  Returns:
    A trimesh object with texture/material applied if available
  """
  mesh_id = mj_model.geom_dataid[geom_idx]

  vert_start = int(mj_model.mesh_vertadr[mesh_id])
  vert_count = int(mj_model.mesh_vertnum[mesh_id])
  face_start = int(mj_model.mesh_faceadr[mesh_id])
  face_count = int(mj_model.mesh_facenum[mesh_id])

  vertices = mj_model.mesh_vert[vert_start : vert_start + vert_count]
  assert vertices.shape == (
    vert_count,
    3,
  ), f"Expected vertices shape ({vert_count}, 3), got {vertices.shape}"

  faces = mj_model.mesh_face[face_start : face_start + face_count]
  assert faces.shape == (
    face_count,
    3,
  ), f"Expected faces shape ({face_count}, 3), got {faces.shape}"

  texcoord_adr = mj_model.mesh_texcoordadr[mesh_id]
  texcoord_num = mj_model.mesh_texcoordnum[mesh_id]

  if texcoord_num > 0:
    if verbose:
      print(f"Mesh has {texcoord_num} texture coordinates")

    texcoords = mj_model.mesh_texcoord[texcoord_adr : texcoord_adr + texcoord_num]
    assert texcoords.shape == (
      texcoord_num,
      2,
    ), f"Expected texcoords shape ({texcoord_num}, 2), got {texcoords.shape}"

    face_texcoord_idx = mj_model.mesh_facetexcoord[face_start : face_start + face_count]
    assert face_texcoord_idx.shape == (face_count, 3), (
      f"Expected face_texcoord_idx shape ({face_count}, 3), got {face_texcoord_idx.shape}"
    )

    # Since the same vertex can have different UVs in different faces,
    # we need to duplicate vertices. Each face will get its own 3 vertices.

    # Duplicate vertices for each face reference.
    # faces.flatten() gives us vertex indices in order:
    # [v0_f0, v1_f0, v2_f0, v0_f1, v1_f1, v2_f1, ...].
    new_vertices = vertices[faces.flatten()]
    assert new_vertices.shape == (
      face_count * 3,
      3,
    ), f"Expected new_vertices shape ({face_count * 3}, 3), got {new_vertices.shape}"

    new_uvs = texcoords[face_texcoord_idx.flatten()]
    assert new_uvs.shape == (
      face_count * 3,
      2,
    ), f"Expected new_uvs shape ({face_count * 3}, 2), got {new_uvs.shape}"

    # Create new faces - now just sequential since vertices are duplicated.
    # [[0, 1, 2], [3, 4, 5], [6, 7, 8], ...]
    new_faces = np.arange(face_count * 3).reshape(-1, 3)
    assert new_faces.shape == (
      face_count,
      3,
    ), f"Expected new_faces shape ({face_count}, 3), got {new_faces.shape}"

    # Create the mesh (process=False to preserve all vertices).
    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

    matid = mj_model.geom_matid[geom_idx]

    if matid >= 0 and matid < mj_model.nmat:
      rgba = mj_model.mat_rgba[matid]
      # mat_texid is 2D (nmat x mjNTEXROLE), try RGB first, then RGBA.
      texid = int(mj_model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
      if texid < 0:
        texid = int(
          mj_model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)]
        )

      if texid >= 0 and texid < mj_model.ntex:
        if verbose:
          print(f"Material has texture ID {texid}")

        tex_width = mj_model.tex_width[texid]
        tex_height = mj_model.tex_height[texid]
        tex_nchannel = mj_model.tex_nchannel[texid]
        tex_adr = mj_model.tex_adr[texid]
        tex_size = tex_width * tex_height * tex_nchannel
        tex_data = mj_model.tex_data[tex_adr : tex_adr + tex_size]
        assert tex_data.shape == (tex_size,), (
          f"Expected tex_data shape ({tex_size},), got {tex_data.shape}"
        )

        # MuJoCo uses OpenGL convention (origin at bottom-left) but GLTF/GLB
        # expects top-left origin, so we flip vertically.
        if tex_nchannel == 1:
          tex_array = tex_data.reshape(tex_height, tex_width)
          tex_array = np.flipud(tex_array)
          image = Image.fromarray(tex_array.astype(np.uint8), mode="L")
        elif tex_nchannel == 3:
          tex_array = tex_data.reshape(tex_height, tex_width, 3)
          tex_array = np.flipud(tex_array)
          image = Image.fromarray(tex_array.astype(np.uint8), mode="RGB")
        elif tex_nchannel == 4:
          tex_array = tex_data.reshape(tex_height, tex_width, 4)
          tex_array = np.flipud(tex_array)
          image = Image.fromarray(tex_array.astype(np.uint8), mode="RGBA")
        else:
          if verbose:
            print(f"Unsupported number of texture channels: {tex_nchannel}")
          image = None

        if image is not None:
          # Set PBR properties for proper rendering:
          # - metallicFactor=0.0: non-metallic (dielectric) material
          # - roughnessFactor=1.0: fully rough (diffuse) surface
          material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=rgba,
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=1.0,
          )
          mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs, material=material)
          if verbose:
            print(f"Applied texture: {tex_width}x{tex_height}, {tex_nchannel} channels")
        else:
          mesh.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.tile(rgba_to_uint8(rgba), (len(new_vertices), 1))
          )
      else:
        if verbose:
          print(f"Material has no texture, using color: {rgba}")
        mesh.visual = trimesh.visual.ColorVisuals(
          vertex_colors=np.tile(rgba_to_uint8(rgba), (len(new_vertices), 1))
        )
    else:
      is_collision = (
        mj_model.geom_contype[geom_idx] != 0 or mj_model.geom_conaffinity[geom_idx] != 0
      )
      color = _DEFAULT_COLLISION_COLOR if is_collision else _DEFAULT_VISUAL_COLOR

      mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(color, (len(new_vertices), 1))
      )
      if verbose:
        print(
          f"No material, using default {'collision' if is_collision else 'visual'} color"
        )

  else:
    if verbose:
      print("Mesh has no texture coordinates")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    matid = mj_model.geom_matid[geom_idx]

    if matid >= 0 and matid < mj_model.nmat:
      rgba = mj_model.mat_rgba[matid]
      mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(rgba_to_uint8(rgba), (len(mesh.vertices), 1))
      )
      if verbose:
        print(f"Applied material color: {rgba}")
    else:
      is_collision = (
        mj_model.geom_contype[geom_idx] != 0 or mj_model.geom_conaffinity[geom_idx] != 0
      )
      color = _DEFAULT_COLLISION_COLOR if is_collision else _DEFAULT_VISUAL_COLOR

      mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(color, (len(mesh.vertices), 1))
      )
      if verbose:
        print(f"Using default {'collision' if is_collision else 'visual'} color")

  assert mesh.vertices.shape[1] == 3, (
    f"Vertices should be Nx3, got {mesh.vertices.shape}"
  )
  assert mesh.faces.shape[1] == 3, f"Faces should be Nx3, got {mesh.faces.shape}"
  assert len(mesh.vertices) > 0, "Mesh has no vertices"
  assert len(mesh.faces) > 0, "Mesh has no faces"

  if verbose:
    print(f"Created mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

  return mesh


def _create_hfield_mesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
  """Create heightfield mesh from MuJoCo hfield data."""
  hfield_id = mj_model.geom_dataid[geom_id]
  nrow = mj_model.hfield_nrow[hfield_id]
  ncol = mj_model.hfield_ncol[hfield_id]
  sx, sy, sz, base = mj_model.hfield_size[hfield_id]

  offset = 0
  for k in range(hfield_id):
    offset += mj_model.hfield_nrow[k] * mj_model.hfield_ncol[k]
  hfield = mj_model.hfield_data[offset : offset + nrow * ncol].reshape(nrow, ncol)

  x = np.linspace(-sx, sx, ncol)
  y = np.linspace(-sy, sy, nrow)
  xx, yy = np.meshgrid(x, y)
  zz = base + sz * hfield

  vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

  faces = []
  for i in range(nrow - 1):
    for j in range(ncol - 1):
      i0 = i * ncol + j
      i1 = i0 + 1
      i2 = i0 + ncol
      i3 = i2 + 1
      faces.append([i0, i2, i1])
      faces.append([i1, i2, i3])
  faces = np.array(faces, dtype=np.int64)
  return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# Dispatch table for primitive shape creation.
_SHAPE_CREATORS = {
  mjtGeom.mjGEOM_SPHERE.value: lambda size: trimesh.creation.icosphere(
    radius=size[0], subdivisions=2
  ),
  mjtGeom.mjGEOM_BOX.value: lambda size: trimesh.creation.box(extents=2.0 * size),
  mjtGeom.mjGEOM_CAPSULE.value: lambda size: trimesh.creation.capsule(
    radius=size[0], height=2.0 * size[1]
  ),
  mjtGeom.mjGEOM_CYLINDER.value: lambda size: trimesh.creation.cylinder(
    radius=size[0], height=2.0 * size[1]
  ),
  mjtGeom.mjGEOM_PLANE.value: lambda size: trimesh.creation.box((20, 20, 0.01)),
}


def _create_ellipsoid_mesh(size: np.ndarray) -> trimesh.Trimesh:
  """Create ellipsoid mesh by scaling a unit sphere."""
  mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
  mesh.apply_scale(size)
  return mesh


def _create_shape_mesh(
  shape_type: int,
  size: np.ndarray,
  rgba: np.ndarray,
  mj_model: mujoco.MjModel | None = None,
  geom_id: int | None = None,
) -> trimesh.Trimesh:
  """Create a mesh for a primitive shape type.

  Args:
    shape_type: MuJoCo geom type (mjtGeom enum value)
    size: Shape size array (interpretation depends on shape_type)
    rgba: RGBA color array (0-1 range)
    mj_model: MuJoCo model (required for HFIELD type)
    geom_id: Geom index (required for HFIELD type)

  Returns:
    Trimesh representation of the shape
  """
  alpha_mode = "BLEND" if rgba[3] < 1.0 else "OPAQUE"
  material = trimesh.visual.material.PBRMaterial(  # type: ignore
    baseColorFactor=rgba,
    metallicFactor=0.0,
    roughnessFactor=1.0,
    alphaMode=alpha_mode,
  )

  if shape_type in _SHAPE_CREATORS:
    mesh = _SHAPE_CREATORS[shape_type](size)
  elif shape_type == mjtGeom.mjGEOM_ELLIPSOID:
    mesh = _create_ellipsoid_mesh(size)
  elif shape_type == mjtGeom.mjGEOM_HFIELD:
    if mj_model is None or geom_id is None:
      raise ValueError("mj_model and geom_id required for HFIELD type")
    mesh = _create_hfield_mesh(mj_model, geom_id)
  else:
    raise ValueError(f"Unsupported shape type: {shape_type}")

  mesh.visual = trimesh.visual.TextureVisuals(material=material)  # type: ignore
  return mesh


def create_primitive_mesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
  """Create a mesh for primitive geom types (sphere, box, capsule, cylinder, plane).

  Args:
    mj_model: MuJoCo model containing geom definition
    geom_id: Index of the geom to create mesh for

  Returns:
    Trimesh representation of the primitive geom
  """
  size = mj_model.geom_size[geom_id]
  geom_type = mj_model.geom_type[geom_id]
  rgba = mj_model.geom_rgba[geom_id].copy()

  # Convert rgba to uint8 for vertex colors.
  rgba_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

  if geom_type == mjtGeom.mjGEOM_SPHERE:
    mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
  elif geom_type == mjtGeom.mjGEOM_BOX:
    mesh = trimesh.creation.box(extents=2.0 * size)
  elif geom_type == mjtGeom.mjGEOM_CAPSULE:
    mesh = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
  elif geom_type == mjtGeom.mjGEOM_CYLINDER:
    mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
  elif geom_type == mjtGeom.mjGEOM_PLANE:
    mesh = trimesh.creation.box((20, 20, 0.01))
  elif geom_type == mjtGeom.mjGEOM_ELLIPSOID:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh.apply_scale(size)
  elif geom_type == mjtGeom.mjGEOM_HFIELD:
    # Which heightfield does this geom use?
    hfield_id = mj_model.geom_dataid[geom_id]
    nrow = mj_model.hfield_nrow[hfield_id]
    ncol = mj_model.hfield_ncol[hfield_id]
    sx, sy, sz, base = mj_model.hfield_size[hfield_id]

    # Compute offset into the flat hfield_data array.
    offset = 0
    for k in range(hfield_id):
      offset += mj_model.hfield_nrow[k] * mj_model.hfield_ncol[k]
    hfield = mj_model.hfield_data[offset : offset + nrow * ncol].reshape(nrow, ncol)

    # MuJoCo heightfield: ncol samples in x direction, nrow samples in y direction.
    # data[r, c] is at position (x[c], y[r]) - columns map to x, rows map to y.
    x = np.linspace(-sx, sx, ncol)
    y = np.linspace(-sy, sy, nrow)
    xx, yy = np.meshgrid(x, y)  # 'xy' indexing: xx[r,c]=x[c], yy[r,c]=y[r]
    # MuJoCo heightfield z formula: z = data * sz (elevation)
    zz = hfield * sz

    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    faces = []
    for r in range(nrow - 1):
      for c in range(ncol - 1):
        # Vertex indices for quad at grid position [r, c]
        i0 = r * ncol + c  # [r, c]     -> (x[c], y[r])
        i1 = i0 + 1  # [r, c+1]   -> (x[c+1], y[r])
        i2 = i0 + ncol  # [r+1, c]   -> (x[c], y[r+1])
        i3 = i2 + 1  # [r+1, c+1] -> (x[c+1], y[r+1])
        # Counter-clockwise winding when viewed from above for upward-pointing normals.
        faces.append([i0, i1, i3])
        faces.append([i0, i3, i2])
    faces = np.array(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Color by height using the same HSV algorithm as color_by_height in heightfield_terrains.py.
    # Normalize z values to [0, 1] range.
    zz_min = zz.min()
    zz_max = zz.max()
    if zz_max > zz_min:
      normalized = (zz - zz_min) / (zz_max - zz_min)
    else:
      normalized = np.full_like(zz, 0.5)

    # HSV color scheme (same as color_by_height in heightfield_terrains.py).
    hue = 0.5 - normalized * 0.45
    saturation = 0.6 - normalized * 0.2
    value = 0.4 + normalized * 0.3

    # HSV to RGB conversion.
    c = value * saturation
    x = c * (1 - np.abs((hue * 6) % 2 - 1))
    m = value - c

    hue_sector = (hue * 6).astype(int) % 6

    r = np.zeros_like(hue)
    g = np.zeros_like(hue)
    b = np.zeros_like(hue)

    mask = hue_sector == 0
    r[mask], g[mask] = c[mask], x[mask]
    mask = hue_sector == 1
    r[mask], g[mask] = x[mask], c[mask]
    mask = hue_sector == 2
    g[mask], b[mask] = c[mask], x[mask]
    mask = hue_sector == 3
    g[mask], b[mask] = x[mask], c[mask]
    mask = hue_sector == 4
    r[mask], b[mask] = x[mask], c[mask]
    mask = hue_sector == 5
    r[mask], b[mask] = c[mask], x[mask]

    r += m
    g += m
    b += m

    # Convert to uint8 vertex colors.
    r_uint8 = (np.clip(r, 0, 1) * 255).astype(np.uint8)
    g_uint8 = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    b_uint8 = (np.clip(b, 0, 1) * 255).astype(np.uint8)
    a_uint8 = np.full_like(r_uint8, 255)

    vertex_colors = np.column_stack(
      [r_uint8.ravel(), g_uint8.ravel(), b_uint8.ravel(), a_uint8.ravel()]
    )

    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
    return mesh
  else:
    raise ValueError(f"Unsupported primitive geom type: {geom_type}")

  # Use ColorVisuals for all primitives to ensure compatibility when merging.
  vertex_colors = np.tile(rgba_uint8, (len(mesh.vertices), 1))
  mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
  return mesh



def merge_geoms(mj_model: mujoco.MjModel, geom_ids: list[int]) -> trimesh.Trimesh:
  """Merge multiple geoms into a single trimesh.

  Args:
    mj_model: MuJoCo model containing geom definitions
    geom_ids: List of geom indices to merge

  Returns:
    Single merged trimesh with all geoms transformed to their local poses
  """
  meshes = []
  for geom_id in geom_ids:
    geom_type = mj_model.geom_type[geom_id]

    if geom_type == mjtGeom.mjGEOM_MESH:
      mesh = mujoco_mesh_to_trimesh(mj_model, geom_id, verbose=False)
    else:
      mesh = create_primitive_mesh(mj_model, geom_id)

    pos = mj_model.geom_pos[geom_id]
    quat = mj_model.geom_quat[geom_id]
    transform = np.eye(4)
    transform[:3, :3] = vtf.SO3(quat).as_matrix()
    transform[:3, 3] = pos
    mesh.apply_transform(transform)
    meshes.append(mesh)

  if len(meshes) == 1:
    return meshes[0]
  return trimesh.util.concatenate(meshes)


def rotation_quat_from_vectors(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
  """Compute quaternion (wxyz format) that rotates from_vec to to_vec.

  Args:
    from_vec: Source vector (3D)
    to_vec: Target vector (3D)

  Returns:
    Quaternion in wxyz format that rotates from_vec to to_vec.
  """
  from_vec = from_vec / np.linalg.norm(from_vec)
  to_vec = to_vec / np.linalg.norm(to_vec)

  if np.allclose(from_vec, to_vec):
    return np.array([1.0, 0.0, 0.0, 0.0])

  if np.allclose(from_vec, -to_vec):
    # 180 degree rotation - pick arbitrary perpendicular axis.
    perp = np.array([1.0, 0.0, 0.0])
    if abs(from_vec[0]) > 0.9:
      perp = np.array([0.0, 1.0, 0.0])
    axis = np.cross(from_vec, perp)
    axis = axis / np.linalg.norm(axis)
    return np.array([0.0, axis[0], axis[1], axis[2]])  # wxyz for 180 deg.

  # Standard quaternion from two vectors.
  cross = np.cross(from_vec, to_vec)
  dot = np.dot(from_vec, to_vec)
  w = 1.0 + dot
  quat = np.array([w, cross[0], cross[1], cross[2]])
  quat = quat / np.linalg.norm(quat)
  return quat


def rotation_matrix_from_vectors(
  from_vec: np.ndarray, to_vec: np.ndarray
) -> np.ndarray:
  """Create rotation matrix that rotates from_vec to to_vec using Rodrigues formula.

  Args:
    from_vec: Source vector (3D)
    to_vec: Target vector (3D)

  Returns:
    3x3 rotation matrix that rotates from_vec to to_vec.
  """
  from_vec = from_vec / np.linalg.norm(from_vec)
  to_vec = to_vec / np.linalg.norm(to_vec)

  if np.allclose(from_vec, to_vec):
    return np.eye(3)

  if np.allclose(from_vec, -to_vec):
    return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

  # Rodrigues rotation formula.
  v = np.cross(from_vec, to_vec)
  s = np.linalg.norm(v)
  c = np.dot(from_vec, to_vec)
  vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def is_fixed_body(mj_model: mujoco.MjModel, body_id: int) -> bool:
  """Check if a body is fixed (welded to world and not attached to mocap).

  A body is considered fixed if it's welded to world AND its kinematic root
  is not a mocap body. This ensures bodies attached to mocap bodies move with them.
  """
  is_weld = mj_model.body_weldid[body_id] == 0
  root_id = mj_model.body_rootid[body_id]
  root_is_mocap = mj_model.body_mocapid[root_id] >= 0
  return is_weld and not root_is_mocap


def get_body_name(mj_model: mujoco.MjModel, body_id: int) -> str:
  """Get body name with fallback to ID-based name.

  Args:
    mj_model: MuJoCo model
    body_id: Body index

  Returns:
    Body name or "body_{body_id}" if name not found.
  """
  body_name = mj_id2name(mj_model, mjtObj.mjOBJ_BODY, body_id)
  if not body_name:
    body_name = f"body_{body_id}"
  return body_name


def merge_sites(mj_model: mujoco.MjModel, site_ids: list[int]) -> trimesh.Trimesh:
  """Merge multiple sites into a single trimesh.

  Args:
    mj_model: MuJoCo model containing site definitions.
    site_ids: List of site indices to merge.

  Returns:
    Single merged trimesh with all sites transformed to their local poses.
  """
  supported_types = {
    mjtGeom.mjGEOM_SPHERE,
    mjtGeom.mjGEOM_BOX,
    mjtGeom.mjGEOM_CAPSULE,
    mjtGeom.mjGEOM_CYLINDER,
    mjtGeom.mjGEOM_ELLIPSOID,
  }
  meshes = []
  for site_id in site_ids:
    site_type = int(mj_model.site_type[site_id])
    if site_type not in supported_types:
      site_type = int(mjtGeom.mjGEOM_SPHERE)
    mesh = _create_shape_mesh(
      shape_type=site_type,
      size=mj_model.site_size[site_id],
      rgba=mj_model.site_rgba[site_id].copy(),
    )
    pos = mj_model.site_pos[site_id]
    quat = mj_model.site_quat[site_id]
    transform = np.eye(4)
    transform[:3, :3] = vtf.SO3(quat).as_matrix()
    transform[:3, 3] = pos
    mesh.apply_transform(transform)
    meshes.append(mesh)

  if len(meshes) == 1:
    return meshes[0]
  return trimesh.util.concatenate(meshes)
