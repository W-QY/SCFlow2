from typing import Optional
import numpy as np
import random
from scipy.spatial.transform import Rotation

from .builder import PIPELINES
from ..pose import load_mesh, eval_rot_error
# from ..multiview import (invert_transform_matrices, get_1_view_TCO_pos_front, get_3_views_TCO_pos_front, get_5_views_TCO_pos_front, 
#                         get_3_views_TCO_pos_sphere, get_6_views_TCO_pos_sphere, get_26_views_TCO_pos_sphere
#                         )

@PIPELINES.register_module()
class PoseJitter:
    def __init__(self, 
                jitter_angle_dis:list,
                jitter_x_dis: list,
                jitter_y_dis: list,
                jitter_z_dis: list,
                jitter_pose_field: list,
                jittered_pose_field: list,
                add_limit: float=None,
                translation_limit: float=None,
                angle_limit: float=None,
                mesh_dir: str=None,
                mesh_diameter: list=None):
        assert isinstance(jitter_angle_dis, (list, tuple))
        assert isinstance(jitter_x_dis, (list, tuple))
        assert isinstance(jitter_y_dis, (list, tuple))
        assert isinstance(jitter_z_dis, (list, tuple))
        assert len(jitter_angle_dis) == 2
        assert len(jitter_z_dis) == 2 and len(jitter_x_dis) == 2 and len(jitter_y_dis) == 2
        self.jitter_angle_dis = jitter_angle_dis
        self.jitter_x_dis = jitter_x_dis
        self.jitter_y_dis = jitter_y_dis
        self.jitter_z_dis = jitter_z_dis
        assert isinstance(jitter_pose_field, (list, tuple))
        assert isinstance(jittered_pose_field, (list, tuple))
        assert len(jittered_pose_field) == len(jitter_pose_field)
        assert 'rotation' in jitter_pose_field[0]
        assert 'translation' in jitter_pose_field[1]
        assert 'rotation' in jittered_pose_field[0]
        assert 'translation' in jittered_pose_field[1]
        self.jitter_pose_field = jitter_pose_field
        self.jittered_pose_field = jittered_pose_field
        self.angle_limit = angle_limit
        self.translation_limit = translation_limit
        self.add_limit = add_limit
        if add_limit is not None:
            self.meshes = load_mesh(mesh_dir)
            mesh_vertices = [mesh.vertices.view(np.ndarray).astype(np.float32) for mesh in self.meshes]
            self.mesh_vertices = [vertices[np.random.choice(vertices.shape[0], 1000)] for vertices in mesh_vertices]
            self.mesh_diameters = mesh_diameter
        
    def jitter(self, rotation, translation, label):
        found_proper_jitter_flag = False
        while not found_proper_jitter_flag:
            angle = [np.random.normal(self.jitter_angle_dis[0], self.jitter_angle_dis[1]) for _ in range(3)]
            delta_rotation = Rotation.from_euler('zyx', angle, degrees=True).as_matrix().astype(np.float32)
            jittered_rotation = np.matmul(delta_rotation, rotation)
            rotation_error = eval_rot_error(rotation[None], jittered_rotation[None])[0]
            if self.angle_limit is not None and rotation_error > self.angle_limit:
                continue

            # translation jitter
            x_noise = np.random.normal(loc=self.jitter_x_dis[0], scale=self.jitter_x_dis[1])
            y_noise = np.random.normal(loc=self.jitter_y_dis[0], scale=self.jitter_y_dis[1])
            z_noise = np.random.normal(loc=self.jitter_z_dis[0], scale=self.jitter_z_dis[1])
            translation_noise = np.array([x_noise, y_noise, z_noise], dtype=np.float32)
            translation_error = np.linalg.norm(translation_noise)
            if self.translation_limit is not None and translation_error > self.translation_limit:
                continue
            jittered_translation = translation + translation_noise
            if self.add_limit is not None:
                verts = self.mesh_vertices[label]
                gt_points = (np.matmul(rotation, verts.T) + translation[:, None]).T
                ref_points = (np.matmul(jittered_rotation, verts.T) + jittered_translation[:, None]).T
                add_error = np.linalg.norm(gt_points - ref_points, axis=-1).mean() / self.mesh_diameters[label]
                if add_error > self.add_limit:
                    continue
            else:
                add_error = None
            return jittered_rotation, jittered_translation, add_error, translation_error, rotation_error

    
    def __call__(self, results):
        rotations, translations = results[self.jitter_pose_field[0]], results[self.jitter_pose_field[1]]
        labels = results['labels']
        k = results['k']
        num_obj = len(rotations)
        if k.ndim == 2:
            k = np.repeat(k[None], num_obj, axis=0)

        jittered_rotations, jittered_translations = [], []
        add_error_list, trans_error_list, rot_error_list = [], [], []
        for i in range(num_obj):
            jittered_rotation, jittered_translation, add_error, rotation_error, translation_error = self.jitter(rotations[i], translations[i], labels[i])
            jittered_translations.append(jittered_translation)
            jittered_rotations.append(jittered_rotation)
            add_error_list.append(add_error)
            rot_error_list.append(rotation_error)
            trans_error_list.append(translation_error)
        add_error = np.array(add_error_list).reshape(-1)
        trans_error = np.array(trans_error_list).reshape(-1)
        rot_error = np.array(rot_error_list).reshape(-1)
        jittered_translations = np.stack(jittered_translations, axis=0)
        jittered_rotations = np.stack(jittered_rotations, axis=0)
        results[self.jittered_pose_field[0]] = jittered_rotations
        results[self.jittered_pose_field[1]] = jittered_translations
        results['init_add_error'] = add_error
        results['init_trans_error'] = trans_error
        results['init_rot_error'] = rot_error
        return results

@PIPELINES.register_module()
class PoseJitterGSO:
    def __init__(self, 
                jitter_angle_dis:list,
                jitter_x_dis: list,
                jitter_y_dis: list,
                jitter_z_dis: list,
                jitter_pose_field: list,
                jittered_pose_field: list,
                add_limit: float=None,
                translation_limit: float=None,
                angle_limit: float=None,
                # mesh_dir: str=None,
                # mesh_diameter: list=None
                ):
        assert isinstance(jitter_angle_dis, (list, tuple))
        assert isinstance(jitter_x_dis, (list, tuple))
        assert isinstance(jitter_y_dis, (list, tuple))
        assert isinstance(jitter_z_dis, (list, tuple))
        assert len(jitter_angle_dis) == 2
        assert len(jitter_z_dis) == 2 and len(jitter_x_dis) == 2 and len(jitter_y_dis) == 2
        self.jitter_angle_dis = jitter_angle_dis
        self.jitter_x_dis = jitter_x_dis
        self.jitter_y_dis = jitter_y_dis
        self.jitter_z_dis = jitter_z_dis
        assert isinstance(jitter_pose_field, (list, tuple))
        assert isinstance(jittered_pose_field, (list, tuple))
        assert len(jittered_pose_field) == len(jitter_pose_field)
        assert 'rotation' in jitter_pose_field[0]
        assert 'translation' in jitter_pose_field[1]
        assert 'rotation' in jittered_pose_field[0]
        assert 'translation' in jittered_pose_field[1]
        self.jitter_pose_field = jitter_pose_field
        self.jittered_pose_field = jittered_pose_field
        self.angle_limit = angle_limit
        self.translation_limit = translation_limit
        self.add_limit = add_limit
        # if add_limit is not None:
        #     self.meshes = load_mesh(mesh_dir)
        #     mesh_vertices = [mesh.vertices.view(np.ndarray).astype(np.float32) for mesh in self.meshes]
        #     self.mesh_vertices = [vertices[np.random.choice(vertices.shape[0], 1000)] for vertices in mesh_vertices]
        #     self.mesh_diameters = mesh_diameter
        
    def jitter(self, rotation, translation, label):
        found_proper_jitter_flag = False
        while not found_proper_jitter_flag:
            angle = [np.random.normal(self.jitter_angle_dis[0], self.jitter_angle_dis[1]) for _ in range(3)]
            delta_rotation = Rotation.from_euler('zyx', angle, degrees=True).as_matrix().astype(np.float32)
            jittered_rotation = np.matmul(delta_rotation, rotation)
            rotation_error = eval_rot_error(rotation[None], jittered_rotation[None])[0]
            if self.angle_limit is not None and rotation_error > self.angle_limit:
                continue

            # translation jitter
            x_noise = np.random.normal(loc=self.jitter_x_dis[0], scale=self.jitter_x_dis[1])
            y_noise = np.random.normal(loc=self.jitter_y_dis[0], scale=self.jitter_y_dis[1])
            z_noise = np.random.normal(loc=self.jitter_z_dis[0], scale=self.jitter_z_dis[1])
            translation_noise = np.array([x_noise, y_noise, z_noise], dtype=np.float32)
            translation_error = np.linalg.norm(translation_noise)
            if self.translation_limit is not None and translation_error > self.translation_limit:
                continue
            jittered_translation = translation + translation_noise
            # if self.add_limit is not None:
            #     verts = self.mesh_vertices[label]
            #     gt_points = (np.matmul(rotation, verts.T) + translation[:, None]).T
            #     ref_points = (np.matmul(jittered_rotation, verts.T) + jittered_translation[:, None]).T
            #     add_error = np.linalg.norm(gt_points - ref_points, axis=-1).mean() / self.mesh_diameters[label]
            #     if add_error > self.add_limit:
            #         continue
            # else:
            add_error = 1.0
            return jittered_rotation, jittered_translation, add_error, translation_error, rotation_error

    
    def __call__(self, results):
        rotations, translations = results[self.jitter_pose_field[0]], results[self.jitter_pose_field[1]]
        labels = results['labels']
        k = results['k']
        num_obj = len(rotations)
        if k.ndim == 2:
            k = np.repeat(k[None], num_obj, axis=0)

        jittered_rotations, jittered_translations = [], []
        add_error_list, trans_error_list, rot_error_list = [], [], []
        for i in range(num_obj):
            jittered_rotation, jittered_translation, add_error, rotation_error, translation_error = self.jitter(rotations[i], translations[i], labels[i])
            jittered_translations.append(jittered_translation)
            jittered_rotations.append(jittered_rotation)
            add_error_list.append(add_error)
            rot_error_list.append(rotation_error)
            trans_error_list.append(translation_error)
        add_error = np.array(add_error_list).reshape(-1)
        trans_error = np.array(trans_error_list).reshape(-1)
        rot_error = np.array(rot_error_list).reshape(-1)
        jittered_translations = np.stack(jittered_translations, axis=0)
        jittered_rotations = np.stack(jittered_rotations, axis=0)
        results[self.jittered_pose_field[0]] = jittered_rotations
        results[self.jittered_pose_field[1]] = jittered_translations
        results['init_add_error'] = add_error
        results['init_trans_error'] = trans_error
        results['init_rot_error'] = rot_error
        return results



# @PIPELINES.register_module()
# class MHPoseJitterv2:
#     def __init__(self, 
#                 jitter_pose_num: int,
#                 jitter_angle_dis: list, 
#                 jitter_t_dis: list, 
#                 jitter_pose_field: list, 
#                 jittered_pose_field: list, 
#                 jitter_type: str='norm',
#                 keep_ori_pose: bool=True,
#                 repeat_jittered_image: bool=True,
#                 # add_limit: float = None, 
#                 translation_limit: float = None, 
#                 angle_limit: float = None, 
#                 ):

#         self.jitter_type = jitter_type
#         self.jitter_pose_num = jitter_pose_num
#         self.jitter_angle_dis = jitter_angle_dis
#         self.jitter_t_dis = jitter_t_dis
#         self.jitter_pose_field = jitter_pose_field
#         self.jittered_pose_field = jittered_pose_field
#         self.repeat_jittered_image = repeat_jittered_image
#         self.keep_ori_pose = keep_ori_pose
#         self.translation_limit = translation_limit
#         self.angle_limit = angle_limit

#     def get_multiview(self, rotation, translation):
#         Pose_matrixes = [np.eye(4, dtype=np.float32)]
#         tmp_matrix = np.eye(4, dtype=np.float32)  
#         tmp_matrix[:3, :3] = rotation.reshape(3, 3) # 4*4
#         tmp_matrix[:3, 3] = translation.copy() / 1000.0
#         tmp_translation = translation.copy() / 1000.0       # 3,
#         if self.jitter_pose_num == 6:
#             Pose_matrixes += get_5_views_TCO_pos_front(tmp_matrix, tmp_translation)
#         elif self.jitter_pose_num == 4:
#             Pose_matrixes += get_3_views_TCO_pos_front(tmp_matrix, tmp_translation)
#         elif self.jitter_pose_num == 2:
#             Pose_matrixes += get_1_view_TCO_pos_front(tmp_matrix, tmp_translation)
#         Pose_matrixes = np.stack(Pose_matrixes).astype(np.float32)
#         Pose_matrixes = invert_transform_matrices(Pose_matrixes) @ tmp_matrix
#         jittered_rotation = Pose_matrixes[:, :3, :3]
#         jittered_translation = Pose_matrixes[:, :3, 3] * 1000.0
#         rotation_error = eval_rot_error(rotation[None], jittered_rotation)
#         translation_error = jittered_translation - tmp_translation
#         return jittered_rotation, jittered_translation


#     def __call__(self, results):
#         rotations, translations = results[self.jitter_pose_field[0]], results[self.jitter_pose_field[1]]
#         jittered_rotations, jittered_translations = [], []
#         num_obj = len(rotations)
#         for i in range(num_obj):
#             # rotation_i, translation_i, label_i = rotations[i], translations[i], labels[i]
#             rotation_i, translation_i = rotations[i], translations[i]
#             jittered_rotation, jittered_translation = self.get_multiview(rotation_i, translation_i)
#             jittered_rotations.append(jittered_rotation)
#             jittered_translations.append(jittered_translation)
#         jittered_translations = np.stack(jittered_translations, axis=0) # (1, mnh_num, 3)
#         jittered_rotations = np.stack(jittered_rotations, axis=0)       # (1, mhp_num, 3, 3)
#         results[self.jittered_pose_field[0]] = jittered_rotations
#         results[self.jittered_pose_field[1]] = jittered_translations
#         return results

@PIPELINES.register_module()
class MHPoseJitter:
    def __init__(self, 
                jitter_pose_num: int,
                jitter_angle_dis: list, 
                jitter_t_dis: list, 
                jitter_pose_field: list, 
                jittered_pose_field: list, 
                jitter_type: str='norm',
                keep_ori_pose: bool=True,
                repeat_jittered_image: bool=True,
                # add_limit: float = None, 
                translation_limit: float = None, 
                angle_limit: float = None, 
                ):

        self.jitter_type = jitter_type
        self.jitter_pose_num = jitter_pose_num
        self.jitter_angle_dis = jitter_angle_dis
        self.jitter_t_dis = jitter_t_dis
        self.jitter_pose_field = jitter_pose_field
        self.jittered_pose_field = jittered_pose_field
        self.repeat_jittered_image = repeat_jittered_image
        self.keep_ori_pose = keep_ori_pose
        self.translation_limit = translation_limit
        self.angle_limit = angle_limit

    def jitter(self, rotation, translation):
        found_proper_jitter_flag = False
        while not found_proper_jitter_flag:
            if self.jitter_type == 'norm':
                angle = [np.random.normal(self.jitter_angle_dis[0], self.jitter_angle_dis[1]) for _ in range(3)]
                x_noise = np.random.normal(loc=self.jitter_t_dis[0], scale=self.jitter_t_dis[1])
                y_noise = np.random.normal(loc=self.jitter_t_dis[0], scale=self.jitter_t_dis[1])
                z_noise = np.random.normal(loc=self.jitter_t_dis[0], scale=self.jitter_t_dis[1])
            elif self.jitter_type == 'uniform':
                angle = [np.random.uniform(-self.jitter_angle_dis[1], self.jitter_angle_dis[1]) for _ in range(3)]
                x_noise = np.random.uniform(low=-self.jitter_t_dis[1], high=self.jitter_t_dis[1])
                y_noise = np.random.uniform(low=-self.jitter_t_dis[1], high=self.jitter_t_dis[1])
                z_noise = np.random.uniform(low=-self.jitter_t_dis[1], high=self.jitter_t_dis[1])
            else:
                print("unsupported jitter type!")

            delta_rotation = Rotation.from_euler('zyx', angle, degrees=True).as_matrix().astype(np.float32)
            jittered_rotation = np.matmul(delta_rotation, rotation)
            rotation_error = eval_rot_error(rotation[None], jittered_rotation[None])[0]
            if self.angle_limit is not None and rotation_error > self.angle_limit:
                continue
            # translation jitter
            translation_noise = np.array([x_noise, y_noise, z_noise], dtype=np.float32)
            translation_error = np.linalg.norm(translation_noise)
            if self.translation_limit is not None and translation_error > self.translation_limit:
                continue
            jittered_translation = translation + translation_noise

            add_error = 1.0
            return jittered_rotation, jittered_translation, add_error, translation_error, rotation_error

    def __call__(self, results):
        rotations, translations = results[self.jitter_pose_field[0]], results[self.jitter_pose_field[1]]
        # labels = results['labels']
        jittered_rotations, jittered_translations = [], []
        num_obj = len(rotations)
        for i in range(num_obj):
            # rotation_i, translation_i, label_i = rotations[i], translations[i], labels[i]
            rotation_i, translation_i = rotations[i], translations[i]
            jittered_rotations_obj_i, jittered_translations_obj_i = [] , []
            if self.keep_ori_pose:
                jittered_rotations_obj_i.append(rotation_i)
                jittered_translations_obj_i.append(translation_i)
                for _ in range(self.jitter_pose_num-1):
                    jittered_rotation, jittered_translation, add_error, rotation_error, translation_error = self.jitter(rotation_i, translation_i)
                    jittered_rotations_obj_i.append(jittered_rotation)
                    jittered_translations_obj_i.append(jittered_translation)
            else:
                for _ in range(self.jitter_pose_num):
                    jittered_rotation, jittered_translation, add_error, rotation_error, translation_error = self.jitter(rotation_i, translation_i)
                    jittered_rotations_obj_i.append(jittered_rotation)
                    jittered_translations_obj_i.append(jittered_translation)
            jittered_rotations_obj_i = np.stack(jittered_rotations_obj_i, axis=0)
            jittered_translations_obj_i = np.stack(jittered_translations_obj_i, axis=0)
            jittered_rotations.append(jittered_rotations_obj_i)
            jittered_translations.append(jittered_translations_obj_i)
        jittered_translations = np.stack(jittered_translations, axis=0) # (1, mnh_num, 3, 3)
        jittered_rotations = np.stack(jittered_rotations, axis=0)       # ()
        results[self.jittered_pose_field[0]] = jittered_rotations
        results[self.jittered_pose_field[1]] = jittered_translations

        # update image keys
        # augmented_images = []
        # all_images = []
        # ori_images = results['img']
        # for i in range(num_obj):
        #     if self.repeat_jittered_image:
        #         all_images.extend([ori_images[i].copy() for _ in range(self.jitter_pose_num)])
        #     else:
        #         all_images.append(ori_images[i].copy())
        # results['ori_img'] = ori_images
        # results['img'] = all_images
        # results['image_filelds'] = ['ori_img', 'augmented_img']
        return results

@PIPELINES.register_module()
class PoseJitterV2(PoseJitter):
    '''
    Different from Class:PoseJitter, jitter depth as a relative ratio according to the original depth
    '''
    def __init__(self, 
                jitter_angle_dis: list, 
                jitter_x_dis: list, 
                jitter_y_dis: list, 
                jitter_z_dis: list, 
                add_limit: list, 
                translation_limit: list, 
                angle_limit: list, 
                mesh_dir: str, 
                mesh_diameter: list, 
                jitter_pose_field: list, 
                jittered_pose_field: list):
        super().__init__(
            jitter_angle_dis, jitter_x_dis, jitter_y_dis, jitter_z_dis, 
            add_limit, translation_limit, angle_limit, 
            mesh_dir, mesh_diameter, 
            jitter_pose_field, jittered_pose_field)
        def __call__(self, results):
            rotations, translations = results[self.jitter_pose_field[0]], results[self.jitter_pose_field[1]]
            labels = results['labels']
            k = results['k']
            num_obj = len(rotations)
            if k.ndim == 2:
                k = np.repeat(k[None], num_obj, axis=0)

            jittered_rotations, jittered_translations = [], []
            add_error_list, trans_error_list, rot_error_list = [], [], []
            for i in range(num_obj):
                found_proper_jitter_flag = False
                while not found_proper_jitter_flag:
                    rotation_i, translation_i, k_i = rotations[i], translations[i], k[i]
                    # rotation jitter
                    angle = [np.random.normal(self.jitter_angle_dis[0], self.jitter_angle_dis[1]) for _ in range(3)]
                    delta_rotation = Rotation.from_euler('zyx', angle, degrees=True).as_matrix().astype(np.float32)
                    jittered_rotation = np.matmul(delta_rotation, rotation_i)
                    rotation_error = eval_rot_error(rotation_i[None], jittered_rotation[None])[0]
                    if self.angle_limit is not None and rotation_error > self.angle_limit:
                        continue

                    # translation xy jitter
                    x_noise = np.random.normal(loc=self.jitter_x_dis[0], scale=self.jitter_x_dis[1])
                    y_noise = np.random.normal(loc=self.jitter_y_dis[0], scale=self.jitter_y_dis[1])
                    # depth jitter
                    ori_depth_i = translation_i[-1]
                    jitter_z_ratio = np.random.uniform(low=self.jitter_z_dis[0]-self.jitter_z_dis[1], high=self.jitter_z_dis[0]+self.jitter_z_dis[1])
                    z_noise = ori_depth_i * jitter_z_ratio - ori_depth_i
                    translation_noise = np.array([x_noise, y_noise, z_noise], dtype=np.float32)
                    translation_error = np.linalg.norm(translation_noise)
                    if self.translation_limit is not None and translation_error > self.translation_limit:
                        continue
                    jittered_translation = translation_i + translation_noise

                    verts = self.mesh_vertices[labels[i]]
                    gt_points = (np.matmul(rotation_i, verts.T) + translation_i[:, None]).T
                    ref_points = (np.matmul(jittered_rotation, verts.T) + jittered_translation[:, None]).T
                    add_error = np.linalg.norm(gt_points - ref_points, axis=-1).mean() / self.mesh_diameters[labels[i]]
                    if self.add_limit is not None and add_error > self.add_limit:
                        continue
                    jittered_translations.append(jittered_translation)
                    jittered_rotations.append(jittered_rotation)
                    add_error_list.append(add_error)
                    rot_error_list.append(rotation_error)
                    trans_error_list.append(translation_error)
                    found_proper_jitter_flag = True
            add_error = np.array(add_error_list).reshape(-1)
            trans_error = np.array(trans_error_list).reshape(-1)
            rot_error = np.array(rot_error_list).reshape(-1)
            jittered_translations = np.stack(jittered_translations, axis=0)
            jittered_rotations = np.stack(jittered_rotations, axis=0)
            results[self.jittered_pose_field[0]] = jittered_rotations
            results[self.jittered_pose_field[1]] = jittered_translations
            results['init_add_error'] = add_error
            results['init_trans_error'] = trans_error
            results['init_rot_error'] = rot_error
            return results
    



@PIPELINES.register_module()
class MultiViewPoseJitterV1:
    def __init__(self, 
                jitter_pose_num:int,
                jitter_pose_field: list, 
                jittered_pose_field: list,
                jitter_rotation_choices:dict,
                jitter_trans_prob:float,
                jitter_x_dis: Optional[list]=None,
                jitter_y_dis: Optional[list]=None,
                jitter_z_dis: Optional[list]=None,
                repeat_jittered_image:bool=False):
        assert jitter_pose_num > 0
        self.jitter_pose_field = jitter_pose_field
        self.jittered_pose_field = jittered_pose_field
        self.jitter_pose_num = jitter_pose_num
        self.jitter_rotation_choices = jitter_rotation_choices
        self.jitter_x_dis = jitter_x_dis
        self.jitter_y_dis = jitter_y_dis
        self.jitter_z_dis = jitter_z_dis
        self.jitter_trans_prob = jitter_trans_prob
        self.repeat_jittered_image = repeat_jittered_image
        self.jitter_rotation_enable = len(jitter_rotation_choices) > 0
    
    def jitter_rotation(self, src_rotation:np.ndarray, jitter_rotation_choices:dict):
        angle = Rotation.from_matrix(src_rotation).as_euler('xyz', degrees=True)
        axis = random.choice(list(jitter_rotation_choices.keys()))
        angle_offset = random.choice(jitter_rotation_choices[axis])
        jittered_angle = angle.copy()
        if axis == 'x':
            jittered_angle[0] = angle[0] + angle_offset
        elif axis == 'y':
            jittered_angle[1] = angle[1] + angle_offset
        else:
            jittered_angle[2] = angle[2] + angle_offset
        jittered_rotation = Rotation.from_euler('xyz', jittered_angle, degrees=True).as_matrix().astype(np.float32)
        jitter_rotation_choices.pop(axis)
        return jittered_rotation

    def __call__(self, results:dict) -> dict:
        rotations, translations = results[self.jitter_pose_field[0]], results[self.jitter_pose_field[1]]
        num_obj = len(rotations)
        jittered_rotations, jittered_translations = [], []
        for i in range(num_obj):
            rotation_i, translation_i = rotations[i], translations[i]
            jittered_rotations_obj_i, jittered_translations_obj_i = [rotation_i], [translation_i]
            jitter_rotation_choices = self.jitter_rotation_choices.copy()
            for _ in range(self.jitter_pose_num):
                # jitter rotation
                jittered_rotation = self.jitter_rotation(rotation_i, jitter_rotation_choices)
                
                if random.random() < self.jitter_trans_prob:
                    # translation jitter
                    x_noise = np.random.normal(loc=self.jitter_x_dis[0], scale=self.jitter_x_dis[1])
                    y_noise = np.random.normal(loc=self.jitter_y_dis[0], scale=self.jitter_y_dis[1])
                    z_noise = np.random.normal(loc=self.jitter_z_dis[0], scale=self.jitter_z_dis[1])
                    translation_noise = np.array([x_noise, y_noise, z_noise], dtype=np.float32)
                    jittered_translation = translation_i + translation_noise
                else:
                    jittered_translation = translation_i
                jittered_rotations_obj_i.append(jittered_rotation)
                jittered_translations_obj_i.append(jittered_translation)
            jittered_rotations.append(np.stack(jittered_rotations_obj_i, axis=0))
            jittered_translations.append(np.stack(jittered_translations_obj_i, axis=0))
        jittered_rotations = np.stack(jittered_rotations, axis=0)
        jittered_translations = np.stack(jittered_translations, axis=0)
        results[self.jittered_pose_field[0]] = jittered_rotations
        results[self.jittered_pose_field[1]] = jittered_translations
        
        # update image keys
        augmented_images = []
        ori_images = results['img']
        for i in range(num_obj):
            if self.repeat_jittered_image:
                augmented_images.extend([ori_images[i].copy() for _ in range(self.jitter_pose_num)])
            else:
                augmented_images.append(ori_images[i].copy())
        results['ori_img'] = ori_images
        results['augmented_img'] = augmented_images
        results['image_filelds'] = ['ori_img', 'augmented_img']
        return results

    
    def update_results(self, results):
        new_images = []
        for img in results['img']:
            new_images.extend([img for _ in range(self.jitter_pose_num+1)])
        results['img'] = new_images
        for key in results.get('annot_fields'):
            if key == 'ref_rotations' or key == 'ref_translations':
                pass
            elif 'masks' in key:
                new_masks = []
                for ele in results[key]:
                    new_masks.extend([ele for _ in range(self.jitter_pose_num+1)])
                results[key] = new_masks
            else:
                results[key] = np.concatenate(
                    [ele[None].repeat(self.jitter_pose_num+1, axis=0) for ele in results[key]], axis=0)
        return results

@PIPELINES.register_module()
class MultiViewPoseJitterV2(PoseJitter):
    def __init__(self, 
                jitter_pose_num: int,
                jitter_angle_dis: list, 
                jitter_x_dis: list, 
                jitter_y_dis: list, 
                jitter_z_dis: list, 
                jitter_pose_field: list, 
                jittered_pose_field: list, 
                keep_ori_pose:bool=False,
                repeat_jittered_image:bool=True,
                add_limit: float = None, 
                translation_limit: float = None, 
                angle_limit: float = None, 
                mesh_dir: str = None, 
                mesh_diameter: list = None):
        super().__init__(
            jitter_angle_dis, jitter_x_dis, jitter_y_dis, jitter_z_dis, 
            jitter_pose_field, jittered_pose_field, 
            add_limit, translation_limit, angle_limit, 
            mesh_dir, mesh_diameter)
        self.jitter_pose_num = jitter_pose_num
        self.repeat_jittered_image = repeat_jittered_image
        self.keep_ori_pose = keep_ori_pose

    def __call__(self, results):
        rotations, translations = results[self.jitter_pose_field[0]], results[self.jitter_pose_field[1]]
        labels = results['labels']
        jittered_rotations, jittered_translations = [], []
        num_obj = len(rotations)
        for i in range(num_obj):
            rotation_i, translation_i, label_i = rotations[i], translations[i], labels[i]
            jittered_rotations_obj_i, jittered_translations_obj_i = [] , []
            if self.keep_ori_pose:
                jittered_rotations_obj_i.append(rotation_i)
                jittered_translations_obj_i.append(translation_i)
                for _ in range(self.jitter_pose_num-1):
                    jittered_rotation, jittered_translation, add_error, rotation_error, translation_error = self.jitter(rotation_i, translation_i, label_i)
                    jittered_rotations_obj_i.append(jittered_rotation)
                    jittered_translations_obj_i.append(jittered_translation)
            else:
                for _ in range(self.jitter_pose_num):
                    jittered_rotation, jittered_translation, add_error, rotation_error, translation_error = self.jitter(rotation_i, translation_i, label_i)
                    jittered_rotations_obj_i.append(jittered_rotation)
                    jittered_translations_obj_i.append(jittered_translation)
            jittered_rotations_obj_i = np.stack(jittered_rotations_obj_i, axis=0)
            jittered_translations_obj_i = np.stack(jittered_translations_obj_i, axis=0)
            jittered_rotations.append(jittered_rotations_obj_i)
            jittered_translations.append(jittered_translations_obj_i)
        jittered_translations = np.stack(jittered_translations, axis=0)
        jittered_rotations = np.stack(jittered_rotations, axis=0)
        results[self.jittered_pose_field[0]] = jittered_rotations
        results[self.jittered_pose_field[1]] = jittered_translations

        # update image keys
        augmented_images = []
        ori_images = results['img']
        for i in range(num_obj):
            if self.repeat_jittered_image:
                augmented_images.extend([ori_images[i].copy() for _ in range(self.jitter_pose_num)])
            else:
                augmented_images.append(ori_images[i].copy())
        results['ori_img'] = ori_images
        results['augmented_img'] = augmented_images
        results['image_filelds'] = ['ori_img', 'augmented_img']
        return results




@PIPELINES.register_module()
class BboxJitter:
    def __init__(self,
                scale_limit,
                shift_limit,
                iof_threshold=0.1,
                p=1.0,
                jitter_bbox_field='gt_bboxes',
                jittered_bbox_field='ref_bboxes',
                mask_field='gt_masks'):
        self.jitter_bbox_field = jitter_bbox_field
        self.jittered_bbox_field = jittered_bbox_field
        self.mask_field = mask_field
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.p = p
        self.iof_threshold = iof_threshold

    def __call__(self, results):
        if random.random() > self.p:
            return results
        bboxes = results.get(self.jitter_bbox_field)
        masks = results.get(self.mask_field)
        w, h = bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]
        cx, cy = (bboxes[:, 2] + bboxes[:, 0])/2, (bboxes[:, 3] + bboxes[:, 1])/2
        num_obj = len(bboxes)
        jittered_bboxes = []
        for i in range(num_obj):
            w_i, h_i, cx_i, cy_i = w[i], h[i], cx[i], cy[i]
            found_valid_jitter_flag = False
            while not found_valid_jitter_flag:
                translate_x = w_i * np.random.uniform(low=-self.shift_limit, high=self.shift_limit)
                translate_y = h_i * np.random.uniform(low=-self.shift_limit, high=self.shift_limit)
                jittered_cx = cx_i + translate_x
                jittered_cy = cy_i + translate_y

                jittered_w = w_i * (np.random.uniform(low=-self.scale_limit, high=self.scale_limit) + 1)
                jittered_h = h_i * (np.random.uniform(low=-self.scale_limit, high=self.scale_limit) + 1)
                
                jittered_x1, jittered_y1 = jittered_cx - jittered_w/2, jittered_cy - jittered_h/2
                jittered_x2, jittered_y2 = jittered_cx + jittered_w/2, jittered_cy + jittered_h/2
                jittered_bbox = np.array(
                    [jittered_x1, jittered_y1, jittered_x2, jittered_y2], dtype=np.float32
                )
                if self.iof_threshold > 0:
                    mask = masks[i]
                    jittered_bbox_mask = np.zeros((mask.height, mask.width), dtype=np.uint8)
                    jittered_bbox_mask[int(jittered_y1):int(jittered_y2), int(jittered_x1):int(jittered_x2)] = 1
                    iof = mask.cal_iof(jittered_bbox_mask)[0]
                    if iof > self.iof_threshold:
                        found_valid_jitter_flag = True
                else:
                    found_valid_jitter_flag = True
            jittered_bboxes.append(jittered_bbox)
        jittered_bboxes = np.stack(jittered_bboxes, axis=0)
        results[self.jittered_bbox_field] = jittered_bboxes
        return results