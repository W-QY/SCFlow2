import os, json, argparse, mmcv, trimesh, cv2
import os.path as osp
import numpy as np
from tqdm import tqdm
from glob import glob

dataset_test_root_all = {'lmo': 'data/lmo/test_bop19', 'tless': 'data/tless/test_primesense', 
                     'tudl': 'data/tudl/test','icbin': 'data/icbin/test', 
                     'itodd': 'data/itodd/test', 'hb': 'data/hb/test_primesense',
                     'ycbv': 'data/ycbv/test_bop19'}

def load_image_list(dataset_test_root, img_list_file):
    with open(img_list_file, 'r') as f:
        img_files = f.readlines()
        img_files = [osp.join(dataset_test_root, x.strip()) for x in img_files]
        img_files = sorted(img_files)
    return img_files

def load_mesh(mesh_path, ext='.ply'):
    if osp.isdir(mesh_path):
        mesh_paths = glob(osp.join(mesh_path, '*'+ext))
        mesh_paths = sorted(mesh_paths)
    else:
        mesh_paths = [mesh_path]
    meshs = [trimesh.load(p) for p in mesh_paths]
    return meshs

def rle_to_binary_mask(rle):
    binary_array = np.zeros(np.prod(rle.get('size')), dtype=bool)
    counts = rle.get('counts')
    start = 0
    for i in range(len(counts)-1):
        start += counts[i] 
        end = start + counts[i+1] 
        binary_array[start:end] = (i + 1) % 2
    binary_mask = binary_array.reshape(*rle.get('size'), order='F')
    return binary_mask

def project_3d_point(pt3d, K, rotation, translation, transform_matrix=None, return_3d=False):
    assert pt3d.ndim == 2, "Only support single object projection"
    if rotation.ndim - translation.ndim ==  1:
        translation = translation[..., None]
    else:
        assert rotation.ndim == translation.ndim
    multi_image = rotation.ndim >= 3
    if transform_matrix is not None:
        assert transform_matrix.ndim == rotation.ndim
    # shape (N, 3, n) or (3, n)
    pts_3d_camera = np.matmul(rotation, pt3d.transpose()) + translation
    # shape (3, n)
    pts_2d = np.matmul(K, pts_3d_camera)
    if transform_matrix is not None:
        pts_2d = np.matmul(transform_matrix, pts_2d)
    pts_2d = pts_2d.transpose()
    pts_2d[..., 0] = pts_2d[..., 0]/ (pts_2d[..., -1] + 1e-8)
    pts_2d[..., 1] = pts_2d[..., 1]/ (pts_2d[..., -1] + 1e-8)
    pts_2d = pts_2d[..., :-1]
    if return_3d:
        if multi_image:
            return pts_2d, pts_3d_camera.transpose((0, 2, 1))
        else:
            return pts_2d, pts_3d_camera.transpose()
    else:
        return pts_2d

def calculate_iou(bbox1, bbox2):    # bbox: (x y w h)
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    w1, h1 = x2_1 - x1_1, y2_1 - y1_1
    w2, h2 = x2_2 - x1_2, y2_2 - y1_2
    inter_x1, inter_y1 = max(x1_1, x1_2), max(y1_1, y1_2)
    inter_x2, inter_y2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area
    
if __name__ == '__main__':

    datasets_all = ['lmo', 'tless', 'tudl', 'icbin', 'itodd', 'hb', 'ycbv']

    for dataset_name in datasets_all:
        # dataset_name = 'lmo'       # lmo tless tudl icbin itodd hb ycbv
        methods_name = 'wdr_init'  # foundpose genflow megapose foundpose+featref                        # change: 0
        model_mesh_dir = 'models_eval'    # lmo: models_eval_13obj others: models_eval
        if dataset_name == 'lmo':
            model_mesh_dir = 'models_eval_13obj'

        data_root = f'data/{dataset_name}'
        dataset_mesh_dir = osp.join(data_root, model_mesh_dir)
        dataset_meshes = load_mesh(dataset_mesh_dir)
        init_mesh_vertices = [mesh.vertices.view(np.ndarray).astype(np.float32) for mesh in dataset_meshes]
        samp_mesh_vertices = [vertices[np.random.choice(vertices.shape[0], 1000)] for vertices in init_mesh_vertices]

        dataset_test_root = dataset_test_root_all[dataset_name]
        image_lists_dir = f'data/{dataset_name}/image_lists/test_bop19.txt'
        image_files = load_image_list(dataset_test_root, image_lists_dir)
        ref_annots_root = f'data/reference_poses/{methods_name}/{dataset_name}'

        # change: 1
        # 00_bop19_list_to_match_pose: cnos_fastsam_bop19_list/sam6d_fastsam_bop19_list/sam6d_sam_bop19_list
        # 01_converted_mask_all: cnos_fastsam_all/sam6d_fast_sam_all/sam6d_sam_all/cosypose_all
        seg_annots_init_root = f'data/reference_masks/01_converted_mask_all/sam6d_fast_sam_all/{dataset_name}'

        # change: 2
        # cnos_per_method/fastsam_per_method/sam_per_method/cosypose_per_method
        seg_annots_save_root = f'data/reference_masks/fastsam_per_method/{methods_name}/{dataset_name}'
        pose_json_tmpl = "{:06d}/scene_gt.json"
        mask_json_tmpl = "{:06d}/scene_seg_info.json"
        camera_json_tmpl = osp.join(dataset_test_root, "{:06}/scene_camera.json")

        sequences = set([p.split(dataset_test_root)[1].split('/')[1] for p in image_files])
        sequences = sorted(list(sequences))
        ref_seq_pose_annots = dict()
        masks_results = dict()
        for sequence in sequences:
            ref_pose_json_path = osp.join(ref_annots_root, pose_json_tmpl.format(int(sequence)))
            ref_mask_json_path = osp.join(seg_annots_init_root, mask_json_tmpl.format(int(sequence)))
            camera_json_path = camera_json_tmpl.format(int(sequence))
            ref_pose_annots = mmcv.load(ref_pose_json_path)
            camera_annots = mmcv.load(camera_json_path)
            mask_annots = mmcv.load(ref_mask_json_path)
            ref_seq_pose_annots[sequence] = dict(pose=ref_pose_annots, camera=camera_annots, mask=mask_annots)


        for img_path in tqdm(image_files):
            _,  seq_name, _, img_name = img_path.rsplit('/', 3)
            img_id = int(osp.splitext(img_name)[0])
            ref_seq_annots = ref_seq_pose_annots[seq_name]

            # load referece pose annots
            if (str(img_id) in ref_seq_annots['pose']) and (str(img_id) in ref_seq_annots['camera']) and (str(img_id) in ref_seq_annots['mask']):
                ref_pose_annots = ref_seq_annots['pose'][str(img_id)]
                camera_annots = ref_seq_annots['camera'][str(img_id)]
                mask_annots = ref_seq_annots['mask'][str(img_id)]
            else:
                print("dataset ", dataset_name, " in ", methods_name, "has no image: ", img_path)
                continue

            ref_obj_num = len(ref_pose_annots)
            assert ref_obj_num != 0, f"Image {img_path} has no references"
            ref_rotations, ref_translations, ref_labels = [], [], []
            for i in range(ref_obj_num):
                obj_id = ref_pose_annots[i]['obj_id']
                ref_rotations.append(np.array(ref_pose_annots[i]['cam_R_m2c'], dtype=np.float32).reshape(3, 3))
                ref_translations.append(np.array(ref_pose_annots[i]['cam_t_m2c'], dtype=np.float32).reshape(-1))
                ref_labels.append(obj_id)
            if len(ref_rotations) == 0:
                raise RuntimeError(f'No valid reference poses in {img_path}')
            ref_rotations, ref_translations = np.stack(ref_rotations, axis=0), np.stack(ref_translations, axis=0)
            ref_labels = np.array(ref_labels, dtype=np.int64) - 1
            k_orig = np.array(camera_annots['cam_K'], dtype=np.float32).reshape(3,3)
            ks = np.repeat(k_orig[None], repeats=ref_obj_num,  axis=0)

            bboxes = []
            for i in range(ref_obj_num):
                ref_rotation, ref_translation = ref_rotations[i], ref_translations[i]
                label, k = ref_labels[i], ks[i]
                points_2d = project_3d_point(samp_mesh_vertices[label], k, ref_rotation, ref_translation)
                points_x, points_y = points_2d[:, 0], points_2d[:, 1]
                left, right = points_x.min(), points_x.max()
                top, bottom = points_y.min(), points_y.max()
                bbox = np.array([left, top, right, bottom], dtype=np.float32)
                bboxes.append(bbox)
            if ref_obj_num > 0:
                bboxes = np.stack(bboxes, axis=0)


            mask_bboxes = []
            for mask_annot in mask_annots:
                bbox_mask = mask_annot['bbox_obj']
                x, y, w, h = bbox_mask[0], bbox_mask[1], bbox_mask[2], bbox_mask[3]
                mask_bboxes.append([x, y, x + w, y + h])
            for i in range(ref_obj_num):
                iou_list = []
                for j in range(len(mask_bboxes)):
                    if mask_annots[j]['obj_id'] != ref_pose_annots[i]['obj_id']:
                        iou_list.append(-1.0)
                    else:
                        iou_list.append(calculate_iou(bboxes[i], mask_bboxes[j]))
                mask_index = iou_list.index(max(iou_list))

                # if ref_pose_annots[i]['score'] != mask_annots[mask_index]['score']:
                #     print("score error!")
                if ref_pose_annots[i]['obj_id'] != mask_annots[mask_index]['obj_id']:
                    print("obj_id error!")

                if int(seq_name) not in masks_results:
                    masks_results[int(seq_name)] = dict()
                if str(img_id) not in masks_results[int(seq_name)]:
                    masks_results[int(seq_name)][str(img_id)] = []
                masks_results[int(seq_name)][str(img_id)].append(
                    dict(
                        bbox_obj=mask_annots[mask_index]['bbox_obj'],
                        obj_id=mask_annots[mask_index]['obj_id'],
                        score=mask_annots[mask_index]['score'],
                        poma_iou=iou_list[mask_index],
                        segmentation=mask_annots[mask_index]['segmentation']
                    )
                )

        for scene_id in masks_results:
            save_path = osp.join(seg_annots_save_root, f"{scene_id:06d}", "scene_seg_info.json")
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            mmcv.dump(masks_results[scene_id], save_path)
        print("method: ", methods_name, "dataset: ", dataset_name, "match over!")