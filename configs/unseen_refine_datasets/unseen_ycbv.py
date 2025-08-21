dataset_root_train = 'data/megapose_dataset'

init_pose_method = 'foundpose'  # foundpose genflow genflow-mh megapose foundpose-fmh foundpose+fm foundpose+featref sam6d foundationpose gigapose+gk
init_mask_method = 'cnos'     # cnos sam fastsam
dataset_root_test = 'data/ycbv'

ref_mask_dir = f'data/reference_masks/{init_mask_method}_per_method/{init_pose_method}/ycbv'
ref_annots_root = f'data/reference_poses/{init_pose_method}/ycbv'


filter_depth = False


train_point_cloud_sample_num = 1024
test_point_cloud_sample_num = 1024

CLASS_NAMES= ('master_chef_can', 'cracker_box',
            'sugar_box', 'tomato_soup_can',
            'mustard_bottle', 'tuna_fish_can',
            'pudding_box', 'gelatin_box',
            'potted_meat_can', 'banana',
            'pitcher_base', 'bleach_cleanser',
            'bowl', 'mug', 'power_drill', 
            'wood_block', 'scissors', 'large_marker',
            'large_clamp', 'extra_large_clamp', 'foam_brick')
normalize_mean = [0., 0., 0., ]
normalize_std = [255., 255., 255.]
image_scale = 256
symmetry_types = { # 1-base
    'cls_13': {'z':0},
    'cls_16': {'x':180, 'y':180, 'z':90},
    'cls_19': {'y':180},
    'cls_20': {'x':180},
    'cls_21': {'x':180, 'y':90, 'z':180}
}
mesh_diameter = [172.16, 269.58, 198.38, 120.66, 199.79, 90.17, 142.58, 114.39, 129.73,
                198.40, 263.60, 260.76, 162.27, 126.86, 230.44, 237.30, 204.11, 121.46,
                183.08, 231.39, 102.92]
file_client_args = dict(backend='disk')


val_pipeline = [
    dict(type='LoadImages', color_type='unchanged', file_client_args=file_client_args),
    # dict(type='LoadMasks', ref_mask_dir=ref_mask_dir),
    dict(type='LoadMasks'),
    dict(type='ComputeBbox', mesh_dir=dataset_root_test + '/models_eval', clip_border=False, filter_invalid=False),
    dict(type='Crop', 
        size_range=(1.1, 1.1),
        crop_bbox_field='gt_bboxes',   # gt_bboxes ref_bboxes  ref_mask_bbox
        clip_border=False,
        pad_val=128),
    dict(type='Resize', img_scale=image_scale, keep_ratio=True),
    dict(type='Pad', size=(image_scale, image_scale), center=True, pad_val=dict(img=(128, 128, 128), mask=0)),
    # dict(type='DepthAug_test', p_rd_block=1.0, p_aug1 = 0.0),
    dict(type='RemapPose', keep_intrinsic=False),
    dict(type='GetPointCloud', filter_depth=filter_depth, filter_point_cloud=False, 
         minimum_points=32, depth_sample_num=test_point_cloud_sample_num,
         filter_rgb=False
         ),
    dict(type='Normalize', mean=normalize_mean, std=normalize_std, to_rgb=True),
    dict(type='ToTensor', stack_keys=[], ),
    dict(type='Collect', 
        annot_keys=[
            'ref_rotations', 'ref_translations',
            'labels','k','ori_k','transform_matrix',
            'depths', 'model_list', 'cloud_list',     #-# add depth 240723
            'gt_masks',
        ],
        meta_keys=(
            'img_path', 'ori_shape', 'img_shape', 'img_norm_cfg', 
            'scale_factor', 'keypoints_3d', 'geometry_transform_mode'),
    ),
]


data = dict(
    samples_per_gpu=8, # 16 # 24
    workers_per_gpu=4,  # 8 # 8
    test_samples_per_gpu=1, # 8
    test=dict(
        type='RefineDataset',
        data_root=dataset_root_test + '/test',
        ref_annots_root=ref_annots_root,
        image_list=dataset_root_test + '/image_lists/test_bop19.txt',
        keypoints_json=dataset_root_test + '/keypoints/bbox.json',
        pipeline=val_pipeline,
        class_names=CLASS_NAMES,
        load_depth=True,            #-# add depth
        load_mask=True,
        load_point_clouds=True,     #-# add point cloud
        filter_invalid_pose=True,      # ---------------------------------------------------
        depth_range=(200, 10000),
        keypoints_num=8,
        mesh_symmetry=symmetry_types,
        meshes_eval=dataset_root_test+'/models_eval',
        mesh_diameter=mesh_diameter,
        mesh_sample_num=test_point_cloud_sample_num,
    ),
)

# renderer setting
model = dict(
    renderer=dict(
        mesh_dir=dataset_root_test + '/models_1023',
        image_size=(image_scale, image_scale),
        shader_type='Phong',
        soft_blending=False,
        render_mask=False,
        render_image=True,
        seperate_lights=True,
        faces_per_pixel=1,
        blur_radius=0.,
        sigma=1e-12,
        gamma=1e-12,
        background_color=(.5, .5, .5),
    ),
)