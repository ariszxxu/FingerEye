from __future__ import annotations
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from pxr import UsdGeom, Sdf, Gf, UsdShade, Usd
import omni.kit.commands
import re
from pathlib import Path
import torch 
import torch.nn.functional as F

FILE_PATH = Path(__file__).resolve()
DIR_PATH = FILE_PATH.parents[6]


def ensure_floor_uv(stage: Usd.Stage, floor_prim_path: str):
    root_prim = stage.GetPrimAtPath(Sdf.Path(floor_prim_path))
    if not root_prim or not root_prim.IsValid():
        return

    mesh_prim = None
    if root_prim.IsA(UsdGeom.Mesh):
        mesh_prim = root_prim
    else:
        for child in stage.Traverse():
            if not str(child.GetPath()).startswith(floor_prim_path):
                continue
            if child.IsA(UsdGeom.Mesh):
                mesh_prim = child
                break

    mesh = UsdGeom.Mesh(mesh_prim)
    path = mesh_prim.GetPath().pathString

    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()

    xformable = UsdGeom.Xformable(mesh_prim)
    world_xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    world_points = [world_xf.Transform(p) for p in points]

    xs = [p[0] for p in world_points]
    ys = [p[1] for p in world_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    range_x = max(max_x - min_x, 1e-6)
    range_y = max(max_y - min_y, 1e-6)

    uvs = []
    for p in world_points:
        u = (p[0] - min_x) / range_x
        v = (p[1] - min_y) / range_y
        uvs.append(Gf.Vec2f(float(u), float(v)))

    primvars_api = UsdGeom.PrimvarsAPI(mesh)

    st_primvar = primvars_api.GetPrimvar("st")
    if not st_primvar:
        st_primvar = primvars_api.CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex,
        )

    st_primvar.Set(uvs)
    st_primvar.SetInterpolation(UsdGeom.Tokens.vertex)

def make_valid_prim_name(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not re.match(r"[A-Za-z_]", token[0]):
        token = "M_" + token
    return token

def get_angle_z_after_x_before(quat_wxyz: np.ndarray) -> float:
    quat_xyzw = np.roll(quat_wxyz, -1)
    R_mat = R.from_quat(quat_xyzw).as_matrix()

    x_before = np.array([1.0, 0.0, 0.0])
    z_after = R_mat[:, 2].reshape(3,)  # Rotated z-axis direction in the original frame.

    cos_theta = np.clip(
        np.dot(x_before, z_after) /
        (np.linalg.norm(x_before) * np.linalg.norm(z_after)),
        -1.0, 1.0
    )
    theta = np.arccos(cos_theta)
    return abs(theta)

def create_skybox_material_from_hdri(stage, mat_prim: str, tex_dir: str):

    files = os.listdir(tex_dir)
    tonemapped = [f for f in files if f.endswith("_TONEMAPPED.jpg")]
    if tonemapped:
        tex_file = tonemapped[0]
    else:
        candidates = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        tex_file = candidates[0]

    tex_path = os.path.join(tex_dir, tex_file)

    material = UsdShade.Material.Define(stage, mat_prim)
    shader_path = mat_prim + "/PreviewSurface"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")

    st_reader_path = mat_prim + "/stReader"
    st_reader = UsdShade.Shader.Define(stage, st_reader_path)
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.TexCoord2f)

    tex = UsdShade.Shader.Define(stage, mat_prim + "/SkyTex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex_path))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")

    st_in = tex.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f)
    st_in.ConnectToSource(st_out)

    brightness = 3.0  
    scale_in = tex.CreateInput("scale", Sdf.ValueTypeNames.Float4)
    scale_in.Set(Gf.Vec4f(brightness, brightness, brightness, 1.0))

    rgb_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)

    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)

    surface_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface_out)

    return material

def create_sky_sphere_mesh(stage, prim_path: str, radius: float = 5.0,
                           n_theta: int = 64, n_phi: int = 32):
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid() and prim.GetTypeName() == "Mesh":
        mesh = UsdGeom.Mesh(prim)
    else:
        if prim.IsValid():
            stage.RemovePrim(prim_path)
        mesh = UsdGeom.Mesh.Define(stage, prim_path)

    points = []
    normals = []
    uvs = []
    counts = []
    indices = []

    for j in range(n_phi + 1):
        v = j / n_phi                # [0,1]
        phi = v * math.pi            # [0,π]
        for i in range(n_theta):
            u = i / n_theta          # [0,1)
            theta = u * 2.0 * math.pi

            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)

            p = Gf.Vec3f(x, y, z)
            points.append(p)

            n = Gf.Vec3f(-x, -y, -z)
            n.Normalize()
            normals.append(n)

            uvs.append(Gf.Vec2f(u, 1-v))

    def idx(i, j):
        return j * n_theta + (i % n_theta)

    for j in range(n_phi):
        for i in range(n_theta):
            i0 = idx(i,     j)
            i1 = idx(i + 1, j)
            i2 = idx(i + 1, j + 1)
            i3 = idx(i,     j + 1)

            counts.append(4)
            indices.extend([i0, i3, i2, i1])

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)

    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation("vertex")

    gprim = UsdGeom.Gprim(mesh.GetPrim())
    gprim.CreateDoubleSidedAttr(True)

    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    st = primvars_api.CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set(uvs)

    return mesh

def augment_image_batch(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        return None

    # ---------- shape and type ----------
    orig_shape = x.shape
    device = x.device
    
    if x.dim() == 5:
        # (N, K, H, W, 3)
        N, K, H, W, C = orig_shape
        is_5d = True
        # flatten to (B, H, W, 3)
        B = N * K
        x_f = x.to(torch.float32).view(B, H, W, C)
    elif x.dim() == 4:
        # (N, H, W, 3)
        N, H, W, C = orig_shape
        is_5d = False
        B = N
        K = None
        x_f = x.to(torch.float32)
    else:
        raise ValueError(f"Unexpected dim {x.dim()}, expect 4 or 5")

    # ---------- 2. Brightness, contrast, and noise (Per-sample / per-camera) ----------
    bc_shape = (N, K, 1, 1, 1) if is_5d else (N, 1, 1, 1)
    
    # Contrast in [0.5, 1.5], brightness in [-25, 25]
    contrast = 0.5 + torch.rand(bc_shape, device=device)
    brightness = (torch.rand(bc_shape, device=device) - 0.5) * 50.0
    
    # Reshape view to match broadcasting
    x_view = x_f.view(bc_shape[0], bc_shape[1] or 1, H, W, C)
    x_view = (x_view - 127.5) * contrast + 127.5 + brightness
    
    # Add Gaussian noise
    x_view = x_view + 40.0 * torch.randn_like(x_view)
    x_f = x_view.view(B, H, W, C)

    # ---------- 3. Local / global blur ----------
    x_ch = x_f.permute(0, 3, 1, 2)  # (B, 3, H, W)
    
    if is_5d:
        # Apply blur to Camera 0 and 2
        target_cam_ids = [c for c in [0, 2] if c < K]  # key is here
        if target_cam_ids:
            blur_k = 7
            pad = blur_k // 2
            sample_idx = torch.arange(N, device=device).unsqueeze(1)
            cam_idx = torch.tensor(target_cam_ids, device=device).view(1, -1)
            flat_idx = (sample_idx * K + cam_idx).reshape(-1)
            
            x_sub = x_ch[flat_idx]
            x_sub = F.pad(x_sub, (pad, pad, pad, pad), mode="reflect")
            x_sub = F.avg_pool2d(x_sub, kernel_size=blur_k, stride=1, padding=0)
            x_ch[flat_idx] = x_sub
    else:
        # Apply small uniform blur in 4D mode
        kernel = 3
        pad = kernel // 2
        x_ch = F.pad(x_ch, (pad, pad, pad, pad), mode="reflect")
        x_ch = F.avg_pool2d(x_ch, kernel_size=kernel, stride=1, padding=0)

    x_f = x_ch.permute(0, 2, 3, 1)  # (B, H, W, 3)

    # ---------- 4. Hue & Saturation ----------
    # Normalize to [0, 1] for HSV conversion
    x_f = (x_f / 255.0).clamp(0.0, 1.0)
    
    eps = 1e-6
    r, g, b = x_f[..., 0], x_f[..., 1], x_f[..., 2]
    maxc, _ = x_f.max(dim=-1)
    minc, _ = x_f.min(dim=-1)
    delta = maxc - minc

    # V & S
    v = maxc
    s = torch.zeros_like(maxc)
    mask_v = maxc > eps
    s[mask_v] = delta[mask_v] / maxc[mask_v]

    # H
    h_tmp = torch.zeros_like(maxc)
    mask_delta = delta > eps
    mask_r = mask_delta & (maxc == r)
    mask_g = mask_delta & (maxc == g)
    mask_b = mask_delta & (maxc == b)

    h_tmp[mask_r] = ((g - b)[mask_r] / (delta[mask_r] + eps)) % 6.0
    h_tmp[mask_g] = ((b - r)[mask_g] / (delta[mask_g] + eps)) + 2.0
    h_tmp[mask_b] = ((r - g)[mask_b] / (delta[mask_b] + eps)) + 4.0
    h = (h_tmp / 6.0) % 1.0

    # Random perturbation parameters
    param_shape = (B, 1, 1)
    
    # Hue: p=0.5, delta ~ U(-0.1, 0.1)
    h_mask = (torch.rand(param_shape, device=device) < 0.5)
    if h_mask.any():
        h_delta = torch.empty(param_shape, device=device).uniform_(-0.1, 0.1)
        h = torch.where(h_mask, (h + h_delta) % 1.0, h)

    # Saturation: p=0.25, factor ~ U(0.5, 2.0)
    s_mask = (torch.rand(param_shape, device=device) < 0.25)
    if s_mask.any():
        s_factor = torch.empty(param_shape, device=device).uniform_(0.5, 2.0)
        s = torch.where(s_mask, (s * s_factor).clamp(0.0, 1.0), s)

    # HSV -> RGB
    c = v * s
    h6 = h * 6.0
    x_inter = c * (1.0 - torch.abs((h6 % 2.0) - 1.0))
    m = v - c

    r_p, g_p, b_p = torch.zeros_like(c), torch.zeros_like(c), torch.zeros_like(c)
    masks = [
        (h6 < 1), (h6 >= 1) & (h6 < 2), (h6 >= 2) & (h6 < 3),
        (h6 >= 3) & (h6 < 4), (h6 >= 4) & (h6 < 5), (h6 >= 5)
    ]
    # Assign values according to interval
    r_p[masks[0]], g_p[masks[0]] = c[masks[0]], x_inter[masks[0]]
    r_p[masks[1]], g_p[masks[1]] = x_inter[masks[1]], c[masks[1]]
    g_p[masks[2]], b_p[masks[2]] = c[masks[2]], x_inter[masks[2]]
    g_p[masks[3]], b_p[masks[3]] = x_inter[masks[3]], c[masks[3]]
    r_p[masks[4]], b_p[masks[4]] = x_inter[masks[4]], c[masks[4]]
    r_p[masks[5]], b_p[masks[5]] = c[masks[5]], x_inter[masks[5]]

    rgb = torch.stack([r_p + m, g_p + m, b_p + m], dim=-1)
    
    # ---------- 5. Restore original shape and type ----------
    rgb = rgb.view(orig_shape)
    return (rgb * 255.0).round().clamp(0, 255).to(torch.uint8)

def quat_to_M6(quat: torch.Tensor, normal: torch.Tensor = None) -> torch.Tensor:
    """
    Inputs
    ------
    quat  (B, 4): quaternion in (w, x, y, z) format
    normal (3,) or None: reference normal (default is [0,0,1])
    
    Output
    ------
    M6 (B, 6): upper-triangle of M = n n^T
                ordered as [M00, M01, M02, M11, M12, M22]
    """
    if normal is None:
        normal = torch.tensor([0., 0., 1.], dtype=quat.dtype, device=quat.device)

    # normalize quaternions to unit length
    quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)

    # get vector part
    w = quat[:, :1]            # (B,1)
    xyz = quat[:, 1:]          # (B,3)

    # expand normal for batch
    v = normal.view(1, 3).expand(quat.shape[0], -1)

    # efficient batched quaternion rotation
    t = torch.cross(xyz, v, dim=-1) * 2
    n = v + w * t + torch.cross(xyz, t, dim=-1)  # (B,3)

    # form nn^T’s unique components
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    M00 = nx * nx
    M01 = nx * ny
    M02 = nx * nz
    M11 = ny * ny
    M12 = ny * nz
    M22 = nz * nz

    return torch.stack([M00, M01, M02, M11, M12, M22], dim=-1)

