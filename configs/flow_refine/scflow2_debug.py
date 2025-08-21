# _base_ = '../refine_datasets/unseen_train.py'


_base_ = '../unseen_refine_datasets/unseen_lmo.py'
# _base_ = '../unseen_refine_datasets/unseen_tless.py'
# _base_ = '../unseen_refine_datasets/unseen_tudl.py'
# _base_ = '../unseen_refine_datasets/unseen_icbin.py'
# _base_ = '../unseen_refine_datasets/unseen_itodd.py'
# _base_ = '../unseen_refine_datasets/unseen_hb.py'
# _base_ = '../unseen_refine_datasets/unseen_ycbv.py'


dataset_root = 'data/ycbv'


mesh_diameter = [172.16, 269.58, 198.38, 120.66, 199.79, 90.17, 142.58, 114.39, 129.73,
                198.40, 263.60, 260.76, 162.27, 126.86, 230.44, 237.30, 204.11, 121.46, 183.08, 231.39, 102.92]
mesh_path = dataset_root+ '/models_eval'

symmetry_types = {}

model = dict(
    type='SCFlow2Refiner',
    cxt_channels=384,   # dgflow: 384, scflow: 128
    h_channels=128,
    seperate_encoder=False,
    add_dense_fusion=False,      # dgflow: True   scflow: False
    cxt_feat_detach=True,       # add in 241122
    max_flow=400.,
    solve_type='reg',  # reg/pnp/kabsch
    filter_invalid_flow=True,
    encoder=dict(
        type='DINOv2Encoder',   # RAFTEncoder DINOv2Encoder
        in_channels=3,
        out_channels=256,
        net_type='basic',       # Basic small, basic, large
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
        type='DINOv2Encoder',   # RAFTEncoder DINOv2Encoder
        in_channels=3,
        out_channels=512,       # dgflow(with fusion): 512 scflow2(wo fusion)/rgb: 256
        net_type='basic',       # Basic small, basic, large
        norm_cfg=dict(type='BN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            # dict(type='Constant', layer=['SyncBatchNorm2d'], val=1, bias=0)
            dict(type='Constant', layer=['InstanceNorm2d'], val=1, bias=0)
        ]),
    decoder=dict(
        type='SCFlow2Decoder',       # DGFlowDecoder     SCFlow2Decoder  SCFlow2DecoderWeightedTs
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=8,            #  ----------------------------------------------------------------- #
        cxt_channels=384,   # dgflow: 384, scflow: 128
        detach_flow=True,
        detach_mask=True,
        detach_pose=True,
        detach_depth_for_xy=True,
        raft3d_rgbd_version=True,   # add in 241231: rgb_version: False
        # debug_ts = True,
        mask_flow=False,
        mask_corr=False,
        pose_head_cfg=dict(
            type='SceneFlowPoseHead',  # dgflow: Raft3DPoseHead/+v2/v3/MHP    scflow: SingleClassPoseHead     # MultiClassPoseHead
            in_channels=19,         # Raft3DPoseHead/v2/v3/MHP: 16,16,17,17; SingleClassPoseHead: 224   
            net_type='Basic',
            rotation_mode='ortho6d',
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            act_cfg=dict(type='ReLU'),
        ),
        corr_lookup_cfg=dict(align_corners=True),
        # gru_type='SeqConv',
        # act_cfg=dict(type='ReLU'),
        ),
    flow_loss_cfg=dict(
        type='SequenceLoss',
        gamma=0.8,
        loss_func_cfg=dict(
            type='RAFTLoss',
            loss_weight=.1,
            max_flow=400.,
        )
    ),
    pose_loss_cfg=dict(
        type='SequenceLoss',
        gamma=0.8,
        loss_func_cfg=dict(
            type='DisentanglePointMatchingLoss',    # DisentanglePointMatchingLoss     # Point2DMatchingLoss
            use_perspective_shape=True,     # add 240723 
            symmetry_types=symmetry_types,
            mesh_diameter=mesh_diameter,
            mesh_path=mesh_path,
            loss_type='l1',
            disentangle_z=True,
            loss_weight=10.0,   # 10.0
        )
    ),
    pose_geo_loss_cfg=dict(
        type='SequenceLoss',
        gamma=0.8,
        loss_func_cfg=dict(
            type='DisentanglePointMatchingLoss',    # DisentanglePointMatchingLoss     # Point2DMatchingLoss
            use_perspective_shape=True,
            symmetry_types=symmetry_types,
            mesh_diameter=mesh_diameter,
            mesh_path=mesh_path,
            loss_type='l1',
            disentangle_z=True,
            loss_weight=10.0,   # 10.0
        )
    ),
    mask_loss_cfg=dict(
        type='SequenceLoss',
        gamma=0.8,
        loss_func_cfg=dict(
            type='L1Loss',
            loss_weight=0.,    # 10.
        )
    ),
    freeze_bn=False,     # False
    freeze_encoder=False,
    train_cfg=dict(
        # rendered_mask_filte=True,
        online_image_renderer=True,         # add in 240726
    ),
    test_cfg=dict(
        # rgb_to_gary=True,   # add in 241124
        iters=8,
        vis_pose_index=-1,
        # vis_index=[1,3,5,7],
        vis_result=False,
        vis_seq_flow=False,
        vis_dir='results/camre_ready/vis_foundpose_ycbv_refined',
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='checkpoints/raft_8x2_100k_flyingchairs.pth'
    )
)


interval = 5000
optimizer_config = dict(grad_clip=dict(max_norm=10.))   # 1/3/5     default = 10


steps = 200000
optimizer = dict(
    type='Adam',
    lr=0.0001,      # 0.0001
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
checkpoint_config = dict(interval=interval, by_epoch=False, max_keep_ckpts=2)
log_config=dict(interval=50, # 50 100
                hooks=[
                    dict(type='TextLoggerHook'),
                    # dict(type='TensorboardImgLoggerHook', interval=100, image_format='HWC')])
                    dict(type='TensorboardImgLoggerHook', interval=50, image_format='HWC')])


# work_dir = 'work_dirs/debug'

work_dir = 'work_dirs/scflow2_wofusion_250624'