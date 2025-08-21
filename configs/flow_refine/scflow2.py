
_base_ = '../unseen_refine_datasets/unseen_lmo.py'
# _base_ = '../unseen_refine_datasets/unseen_ycbv.py'


dataset_root = 'data/ycbv'



train_unseen=True

if dataset_root == 'data/ycbv':
    symmetry_types = { # 1-base
        'cls_13': {'z':0},
        'cls_16': {'x':180, 'y':180, 'z':90},
        'cls_19': {'y':180},
        'cls_20': {'x':180},
        'cls_21': {'x':180, 'y':90, 'z':180}
    }
    mesh_diameter = [172.16, 269.58, 198.38, 120.66, 199.79, 90.17, 142.58, 114.39, 129.73,
                    198.40, 263.60, 260.76, 162.27, 126.86, 230.44, 237.30, 204.11, 121.46, 183.08, 231.39, 102.92]
    mesh_path = dataset_root+ '/models_eval'

elif dataset_root == 'data/lm':
    symmetry_types = { # 1-base
        'cls_8': {'x':180, 'y':180, 'z':180},
        'cls_9': {'z':180},
    }
    mesh_diameter =[102.099, 247.506, 167.355, 172.492, 201.404, 154.546, 124.264, 
                    261.472, 108.999, 164.628, 175.889, 145.543, 278.078, 282.601, 212.358]
    mesh_path = dataset_root+ '/models_eval_13obj'

elif dataset_root == 'data/itodd':
    symmetry_types = {}
    mesh_diameter = [64.0944, 51.4741, 142.15, 139.379, 158.583, 85.3086, 38.5388, 68.884, 94.8011, 55.7152, 140.121, 107.703, 128.059, 102.883, 
                     114.191, 193.148, 77.7869, 108.482, 121.383, 122.019, 171.23, 267.47, 56.9323, 65, 48.5103, 66.8026, 55.7315, 24.0832,]
    mesh_path = dataset_root + '/models_eval'

if train_unseen:
    symmetry_types = {}

model = dict(
    type='SCFlow2Refiner',
    cxt_channels=384,
    h_channels=128,
    seperate_encoder=False,
    cxt_feat_detach=True,
    max_flow=400.,
    solve_type='reg',
    add_dense_fusion=True,
    filter_invalid_flow=True,
    encoder=dict(
        type='DINOv2Encoder',
        in_channels=3,
        out_channels=256,
        net_type='basic',
        norm_cfg=dict(type='IN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['InstanceNorm2d'], val=1, bias=0)
        ]),
    cxt_encoder=dict(
        type='SCFlow2Decoder',
        in_channels=3,
        out_channels=256,
        net_type='basic',
        norm_cfg=dict(type='BN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['SyncBatchNorm2d'], val=1, bias=0)
        ]),
    decoder=dict(
        type='SCFlow2Decoder',
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=8,
        cxt_channels=384,
        detach_flow=True,
        detach_mask=True,
        detach_pose=True,
        detach_depth_for_xy=True,
        depth_based_upsample=False,
        mask_flow=False,
        mask_corr=False,
        pose_head_cfg=dict(
            type='SceneFlowPoseHead',
            in_channels=16,
            net_type='Basic',
            rotation_mode='ortho6d',
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            act_cfg=dict(type='ReLU'),
        ),
        corr_lookup_cfg=dict(align_corners=True),
        gru_type='SeqConv',
        act_cfg=dict(type='ReLU')),
    freeze_bn=False,     # False
    freeze_encoder=False,
    train_cfg=dict(
        rendered_mask_filte=True,
        online_image_renderer=train_unseen,         # add in 240726
    ),
    test_cfg=dict(
        iters=8,
    ),
    # init_cfg=dict(
    #     type='Pretrained',
    #     checkpoint='work_dirs/dgflow_unseen/scflow2_rawbbox/dinov2b_rawbbox_iter_145000.pth'
    # )
)


interval = 5000
optimizer_config = dict(grad_clip=dict(max_norm=10.))   # 1/3/5     default = 10
steps = 200000
optimizer = dict(
    type='Adam',
    lr=0.0001,
    betas=(0.5, 0.999),
    eps=1e-06,
    weight_decay=0.0,
    amsgrad=False,
)
lr_config = dict(
    policy='CosineAnnealing',
    warmup_ratio=0.001,
    warmup_iters=1000,
    warmup='linear',
    min_lr=0,
)

evaluation=dict(interval=interval,         # evaluation
                metric={
                    # 'bop':[],
                    'auc':[],
                    'add':[0.05, 0.10, 0.20, 0.50]},
                save_best='average/add_10',
                rule='greater'
            )
runner = dict(type='IterBasedRunner', max_iters=steps)
num_gpus = 1
checkpoint_config = dict(interval=interval, by_epoch=False)
log_config=dict(interval=50,
                hooks=[
                    dict(type='TextLoggerHook'),
                    dict(type='TensorboardImgLoggerHook', interval=50, image_format='HWC')])

work_dir = 'work_dirs/debug'
