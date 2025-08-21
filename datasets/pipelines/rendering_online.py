import torch, trimesh, logging
from torch import nn
import numpy as np
from glob import glob
from os import path as osp
from iopath.common.file_io import PathManager
from pytorch3d.structures import join_meshes_as_batch, Meshes
from pytorch3d.renderer import (
    PointLights, PerspectiveCameras,BlendParams, AmbientLights,
    MeshRasterizer, RasterizationSettings, Textures, TexturesUV,
    HardPhongShader, SoftPhongShader, HardGouraudShader, SoftGouraudShader, SoftSilhouetteShader,
    HardFlatShader)
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from pytorch3d.io.ply_io import MeshPlyFormat
from torchvision.utils import save_image
from pytorch3d.io import load_objs_as_meshes
import random
logging.getLogger('trimesh').setLevel(logging.ERROR)

def cameras_from_opencv_projection(
    R: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,
    device,
) -> PerspectiveCameras:
    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Screen to NDC conversion:
    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer, as well as
    # the transformation function `get_ndc_to_screen_transform`.
    scale = (image_size_wh.to(R).min(dim=1, keepdim=True)[0] - 1) / 2.0
    scale = scale.expand(-1, 2)
    c0 = (image_size_wh - 1) / 2.0

    # Get the PyTorch3D focal length and principal point.
    focal_pytorch3d = focal_length / scale
    p0_pytorch3d = -(principal_point - c0) / scale

    # For R, T we flip x, y axes (opencv screen space has an opposite orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    # The original code clone the R and T, which will cause R and T requires_grad = False
    # TODO figure out why clone the R and T
    R_pytorch3d = R.clone().permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] = R_pytorch3d[:, :, :2] * -1
    T_pytorch3d[:, :2] *= -1

    return PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
        image_size=image_size,
        device=device
    )



def load_mesh(f, path_manager=None):
    if path_manager is None:
        path_manager = PathManager()
    mesh = MeshPlyFormat().read(f, include_textures=True, device='cpu', path_manager=path_manager)
    return mesh

def load_ply_mesh(f, device='cpu', path_manager=None):
    if path_manager is None:
        path_manager = PathManager()
    mesh = MeshPlyFormat().read(f, include_textures=True, device=device, path_manager=path_manager)
    return mesh

def load_obj_mesh(f, device, dataset_type):
    if dataset_type == 'MegaPose-ShapeNetCore':
        mesh = load_objs_as_meshes(f, device=device, load_textures=True, create_texture_atlas=True, texture_wrap="clamp")
    else:
        mesh = load_objs_as_meshes(f, device=device, load_textures=True)
    return mesh

def vertices_scale(vertices):
    xmin, xmax = float(vertices[:, 0].min()), float(vertices[:, 0].max())
    ymin, ymax = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    zmin, zmax = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    scale = max(max(xmax - xmin, ymax - ymin), zmax - zmin) #  / 2.0
    vertices[:, 0] -= (xmax + xmin) / 2.0
    vertices[:, 1] -= (ymax + ymin) / 2.0
    vertices[:, 2] -= (zmax + zmin) / 2.0
    vertices[:, :3] /= scale 
    # vertices[:, :3] *= 0.29777
    return vertices

def load_glb_mesh(path, mesh_scale_data, device):
    # path = 'data/megapose_dataset/FoundationPose-Objaverse/Objaverse_objects/glbs/000-120/3887d194684943c98384acb97024510b.glb'
    mesh = trimesh.load(path, force='mesh')
    vertices = np.array(mesh.vertices, dtype=np.float32)
    vertices = vertices_scale(vertices)
    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.int64)
    if hasattr(mesh.visual, 'material'):
        material = mesh.visual.material
        if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
            texture_image = material.baseColorTexture
            texture_image = torch.tensor(np.array(texture_image) / 255.0, dtype=torch.float32)
            uv_coords = torch.tensor(mesh.visual.uv, dtype=torch.float32)
            textures = TexturesUV(maps=[texture_image], faces_uvs=[faces], verts_uvs=[uv_coords])
        elif hasattr(material, 'image') and material.image is not None:
            texture_image = material.image
            texture_image = torch.tensor(np.array(texture_image) / 255.0, dtype=torch.float32)
            uv_coords = torch.tensor(mesh.visual.uv, dtype=torch.float32)
            textures = TexturesUV(maps=[texture_image], faces_uvs=[faces], verts_uvs=[uv_coords])
        else:
            base_color_factor = material.baseColorFactor[:3] if material.baseColorFactor is not None else [1.0, 1.0, 1.0]
            texture_image = torch.tensor(base_color_factor, dtype=torch.float32).view(1, 1, 3).expand(1024, 1024, 3)
            uv_coords = torch.tensor(mesh.visual.uv, dtype=torch.float32)
            textures = TexturesUV(maps=[texture_image], faces_uvs=[faces], verts_uvs=[uv_coords])
    elif hasattr(mesh.visual, 'vertex_colors'):
        vertex_colors = mesh.visual.vertex_colors[:, :3]
        vertex_colors = torch.tensor(vertex_colors / 255.0, dtype=torch.float32)
        textures = Textures(verts_rgb=[vertex_colors])
    model = Meshes(verts=[vertices], faces=[faces], textures=textures)
    model.scale_verts_(1000 * mesh_scale_data)
    model = model.to(device)
    del mesh, vertices, faces, textures
    torch.cuda.empty_cache()
    return model

shader_mapping = {
    'Phong': dict(hard=HardPhongShader, soft=SoftPhongShader),
    'Gouraud':dict(hard=HardGouraudShader, soft=SoftGouraudShader),
    'Flat':dict(hard=HardFlatShader),
}

class RendererOnline(nn.Module):
    def __init__(self,
                image_size, # (H, W)
                shader_type='Phong',
                soft_blending=True,
                render_mask=True,
                render_image=True,
                faces_per_pixel=1,
                blur_radius=0.,
                sigma=1e-4,
                gamma=1e-4,
                bin_size=-1,      # None 
                default_lights=True,
                seperate_lights=False,
                background_color=(0.5, 0.5, 0.5),
                ):
        super(RendererOnline, self).__init__()
       
        assert shader_type in shader_mapping.keys()
        
        self.image_size = image_size
        self.render_mask = render_mask
        self.render_image = render_image
        self.shader_type = shader_type
        self.blending = 'soft' if soft_blending else 'hard'
        self.faces_per_pixel = faces_per_pixel
        self.blur_radius = blur_radius
        self.gamma = gamma
        self.sigma = sigma
        self.background_color = background_color
        self.shader = shader_mapping[self.shader_type][self.blending]
        # self.load_meshes(test_mesh_dir)
        self.default_lights = default_lights
        self.seperate_lights = seperate_lights
        self.bin_size = bin_size

    def to(self, device):
        self._init_renderer(device)
        # if self.image_renderer is not None:
        #     self.image_renderer.to(device)
        # if self.mask_renderer is not None:
        #     self.mask_renderer.to(device)
        # for k in self.meshes:
        #     self.meshes[k] = self.meshes[k].to(device)
    
    # def load_meshes(self, mesh_dir, ext='.ply'):
    #     if osp.isdir(mesh_dir):
    #         mesh_paths = glob(osp.join(mesh_dir, '*'+ext))
    #     else:
    #         mesh_paths = [mesh_dir]
    #     mesh_paths = sorted(mesh_paths)
    #     self.meshes = dict()
    #     for mesh_path in mesh_paths:
    #         obj_label = int(osp.basename(mesh_path).split('.')[0].split('_')[-1])-1
    #         self.meshes[obj_label] = load_mesh(mesh_path)
            
    def _init_renderer(self, device):
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=self.blur_radius,
            faces_per_pixel=self.faces_per_pixel,
            bin_size=self.bin_size,
        )
        self.blend_params = BlendParams(gamma=self.gamma, sigma=self.sigma, background_color=self.background_color)
        if self.render_image:
            self.image_renderer = MeshRendererWithFragments(
                rasterizer=MeshRasterizer(
                    cameras=None,
                    raster_settings=self.raster_settings
                ),
                shader = self.shader(
                    cameras=None,
                    blend_params=self.blend_params,
                    device=device
                )
            )
        else:
            self.image_renderer = None
       
        if self.render_mask:
            self.mask_renderer = MeshRendererWithFragments(
                rasterizer=MeshRasterizer(
                    cameras=None,
                    raster_settings=self.raster_settings
                ),
                shader=SoftSilhouetteShader(
                    blend_params=self.blend_params,
                )
            )
        else:
            self.mask_renderer = None
    
    def forward(self, rotations, translations, internel_k, paths, mesh_scale_datas):
        assert rotations.size(0) == translations.size(0) == internel_k.size(0) == len(paths)
        num_images = rotations.size(0)
        device = rotations.device
        meshes_to_render_list = []
        for i in range(len(paths)):
            path = paths[i]
            dataset_type = path.split('/')[2]
            if dataset_type == 'FoundationPose-Objaverse':
                try:
                    mesh_scale_data = mesh_scale_datas[i]
                    mesh = load_glb_mesh(path, mesh_scale_data, device)
                except:
                    return None
            else:
                # obj
                try:
                    mesh = load_obj_mesh([path], device, dataset_type)
                except ValueError as e:
                    return None
                except Exception as e:
                    return None
                mesh.scale_verts_(100)
            # ply
            # mesh = load_ply_mesh(img_meta['mesh_ply_paths'][0], device)
            meshes_to_render_list.append(mesh)
        if rotations.size(0) == 1 and len(meshes_to_render_list) == 1 and rotations.dim() == 4:  #-# added in 250218 for MHposes
            rotations = rotations[0]
            translations = translations[0]
            for i in range(rotations.size(0)-1):
                meshes_to_render_list.append(mesh)

        del mesh
        meshes_to_render = join_meshes_as_batch(meshes_to_render_list, include_textures=True)
        verts_list = meshes_to_render.verts_list()
        zbuf_list = []
        for verts, rotation, translation in zip(verts_list, rotations, translations):
            points_3d = torch.matmul(rotation, verts.transpose(0, 1).to(rotation.device)) + translation[:, None]
            zbuf_list.append(points_3d[-1, :])
        zbuf = torch.cat(zbuf_list)
        zfar, znear = torch.max(zbuf).item(), torch.min(zbuf).item()
        zfar, znear = (zfar // 100 + 1) * 100., (znear // 100) * 100.
        
        image_size = torch.tensor(self.image_size, device=device)[None].expand(num_images, 2)
        # initialize cameras
        cameras = cameras_from_opencv_projection(
            R=rotations, 
            tvec=translations, 
            camera_matrix=internel_k, 
            image_size=image_size, 
            device=device)


        # lights = AmbientLights(device=device, ambient_color=[[0.9, 0.9, 0.9]])

        # if dataset_type == 'MegaPose-ShapeNetCore':
        # if True:
        #     lights = AmbientLights(device=device, ambient_color=[[0.9, 0.9, 0.9]])
        # else:
        if not self.default_lights:
            # for ITODD
            if self.seperate_lights:
                znear_list = torch.stack([z.min() for z in zbuf_list])
                znear_list = torch.maximum(znear_list - 400, torch.zeros_like(znear_list))
                loc = torch.stack([torch.zeros_like(znear_list), torch.zeros_like(znear_list), znear_list], axis=-1)
                loc = (rotations @ loc[..., None]).view(-1, 3)
            else:
                loc = torch.tensor((0., 0. ,znear/4)).to(translations.device).view(1,-1,1) # flipped Z
                loc = (rotations@loc).view(-1,3)
            lights = PointLights(diffuse_color=((.5, .5, .5),), ambient_color=((.8, .8, .8),), specular_color=((1., 1., 1.,),), location=loc, device=device)
        else:
            if self.seperate_lights:
                znear_list = torch.stack([z.min() for z in zbuf_list])
                znear_list = torch.maximum(znear_list - 400, torch.zeros_like(znear_list))
                loc = torch.stack([torch.zeros_like(znear_list), torch.zeros_like(znear_list), znear_list], axis=-1)
                loc = (rotations @ loc[..., None]).view(-1, 3)
                lights = PointLights(location=loc, device=device)
            else:
                lights = PointLights(device=device)


        outputs = dict()
        if self.image_renderer is not None:
            try:
                images, fragments = self.image_renderer(meshes_to_render, cameras=cameras, znear=znear, zfar=zfar, lights=lights)
            except ValueError as e:
                return None
            except Exception as e:
                return None
            outputs.update(
                images = images,
                fragments = fragments,
            )
        if self.mask_renderer is not None:
            masks, fragments = self.mask_renderer(meshes_to_render, cameras=cameras, znear=znear, zfar=zfar)
            outputs.update(
                masks = masks,
            )
            if 'fragments' not in outputs:
                outputs.update(
                    fragments = fragments
                )
        del meshes_to_render
        del meshes_to_render_list
        return outputs
