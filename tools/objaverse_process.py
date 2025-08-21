import os, cv2, json, glob, shutil, multiprocessing
import numpy as np
from tqdm import tqdm
import os.path as osp
from pycocotools import mask as mask_util

glcam_in_cvcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]]).astype(float)

def get_camera_intrinsics(json_path, W, H):
    with open(json_path,'r') as ff:
        camera_params = json.load(ff)
    world_in_glcam = np.array(camera_params['cameraViewTransform']).reshape(4,4).T
    cam_in_world = np.linalg.inv(world_in_glcam)@glcam_in_cvcam
    world_in_cam = np.linalg.inv(cam_in_world)
    focal_length = camera_params["cameraFocalLength"]
    horiz_aperture = camera_params["cameraAperture"][0]
    vert_aperture = H / W * horiz_aperture
    focal_y = H * focal_length / vert_aperture
    focal_x = W * focal_length / horiz_aperture
    center_y = H * 0.5
    center_x = W * 0.5
    fx, fy, cx, cy = focal_x, focal_y, center_x, center_y
    K = np.eye(3)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy
    return K, world_in_cam

# convert mask
def extract_mask(mask_array, n):
    binary_mask = (mask_array == n).astype(np.uint8)
    return binary_mask
def rle_encode_coco(mask):
    """
    Encodes a binary mask using COCO's RLE format.
    :param mask: 2D numpy array where '1's represent the object
    :return: RLE in COCO format {'counts': list, 'size': [height, width]}
    """
    pixels = mask.T.flatten()
    # changes = np.diff(pixels, prepend=pixels[0], append=1)
    changes = np.diff(pixels, prepend=0, append=0)
    run_lengths = np.where(changes != 0)[0]
    counts = np.diff(np.concatenate([[0], run_lengths]))
    return {'counts': counts.tolist(), 'size': mask.shape}

def rle_to_binary_mask(rle):
    """Converts a COCOs run-length encoding (RLE) to binary mask.

    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get('size')), dtype=bool)
    counts = rle.get('counts')
    start = 0
    for i in range(len(counts)-1):
        start += counts[i] 
        end = start + counts[i+1] 
        binary_array[start:end] = (i + 1) % 2
    binary_mask = binary_array.reshape(*rle.get('size'), order='F')
    return binary_mask

# convert Rotation matrix
def normalizeRotation(pose):
    new_pose = pose.copy()
    scales = np.linalg.norm(pose[:3,:3],axis=0)
    new_pose[:3,:3] /= scales.reshape(1,3)
    return new_pose, scales[0]

def get_point_cloud_from_depth(depth, K):
    cam_fx, cam_fy, cam_cx, cam_cy = K[0,0], K[1,1], K[0,2], K[1,2]

    im_H, im_W = depth.shape
    xmap = np.array([[i for i in range(im_W)] for j in range(im_H)])
    ymap = np.array([[j for i in range(im_W)] for j in range(im_H)])

    pt2 = depth.astype(np.float32)
    pt0 = (xmap.astype(np.float32) - cam_cx) * pt2 / cam_fx
    pt1 = (ymap.astype(np.float32) - cam_cy) * pt2 / cam_fy

    cloud = np.stack([pt0,pt1,pt2]).transpose((1,2,0))
    return cloud


def process_one_scene(scene_dir, scene_id):
    scene_fold = glob.glob(scene_dir + '/scene-*')[0]
    state_json = scene_dir + '/states.json'
    if scene_id == 0:
        scene_sub = '/RenderProduct_Replicator'
    elif scene_id ==1:
        scene_sub = '/RenderProduct_Replicator_01'
    camera_json = scene_fold + scene_sub +  '/camera_params/camera_params_000000.json'
    mask_path = scene_fold + scene_sub + '/instance_segmentation/instance_segmentation_000000.png'
    bbox_path = scene_fold + scene_sub + '/bounding_box_2d_loose/bounding_box_2d_loose_000000.npy'
    depth_path = scene_fold + scene_sub + '/distance_to_image_plane/distance_to_image_plane_000000.npy'
    mask_mapping_json = scene_fold + scene_sub + '/instance_segmentation/instance_segmentation_mapping_000000.json'
    bbox_mapping_json = scene_fold + scene_sub + '/bounding_box_2d_loose/bounding_box_2d_loose_prim_paths_000000.json'
    obj_id_json_path = 'data/megapose_dataset/FoundationPose-Objaverse/objaverse_name_index_id.json'
    image_path = scene_fold + scene_sub + '/rgb/rgb_000000.png'

    # load
    with open(state_json, 'r') as f:
        state_list = json.load(f)
    with open(mask_mapping_json, 'r') as f:
        mask_mapping_list = json.load(f)
    with open(bbox_mapping_json, 'r') as f:
        bbox_mapping_list = json.load(f)
    with open(obj_id_json_path, 'r') as f:
        obj_id_list = json.load(f)
    bbox_data = np.load(bbox_path)
    depth_data = np.load(depth_path)

    out_camera_dict, out_gtinfo_list, out_gt_list, out_mask_visib_list = dict(), [], [], dict()
    W, H = 640, 480
    intrinsics, world_in_cam = get_camera_intrinsics(camera_json, W, H)

    # rgb mv, depth process and process:
    target_rgb_path = os.path.join(scene_dir, f'00000{scene_id}.rgb.png')
    if not os.path.exists(target_rgb_path):
        shutil.move(image_path, target_rgb_path)
    depth_scaled = np.floor(depth_data * 1e4).astype(np.uint16)     # init 1e4
    cv2.imwrite(os.path.join(scene_dir, f'00000{scene_id}.depth.png'), depth_scaled)
    
    # camera json process and save:
    out_camera_dict["cam_K"] = intrinsics.flatten().tolist()
    out_camera_dict["cam_R_w2c"] = world_in_cam[:3, :3].flatten().tolist()
    out_camera_dict["cam_t_w2c"] = world_in_cam[:3, 3].flatten().tolist()
    out_camera_dict["depth_scale"] = 0.1
    with open(os.path.join(scene_dir, f'00000{scene_id}.camera.json'), 'w') as f:
        json.dump(out_camera_dict, f, indent=None, separators=(',', ': '))
    
    # read mask and process json info
    mask_array = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    unique_values, counts = np.unique(mask_array, return_counts=True)

    valid_mask_ids = unique_values[(unique_values != 0) & (unique_values != 4)]
    valid_counts = counts[(unique_values != 0) & (unique_values != 4)]
    valid_mask_num = len(valid_mask_ids)
    obj_nums = valid_mask_num           # include bbox

    valid_obj_index = 0
    for i in range(obj_nums):
        # get name and id
        mask_id = valid_mask_ids[i]
        px_count_visib = valid_counts[i]
        if not str(mask_id) in mask_mapping_list:
            continue
        mesh_name = mask_mapping_list[str(mask_id)].split('objaverse_')[1].split('/')[0]
        obj_id = obj_id_list[mesh_name]

        #-- mask process
        binary_mask = extract_mask(mask_array, mask_id)
        mask_rle = rle_encode_coco(binary_mask)
        out_mask_visib_list[str(valid_obj_index)] = mask_rle
        # verify:
        load_mask = rle_to_binary_mask(mask_rle).astype(np.uint8)
        if not np.array_equal(load_mask, binary_mask):
            print("error: ", mask_path, "mask_id: ", mask_id)

        #-- debug cloud
        # depth_scaled = np.floor(depth_data * 1e3).astype(np.uint16)  # -*-----------------
        # choose = binary_mask.astype(np.float32).flatten().nonzero()[0]
        # obj_cloud = get_point_cloud_from_depth(depth_scaled, intrinsics)
        # obj_cloud = obj_cloud.reshape(-1, 3)[choose, :]
        # np.save('point_cloud/src.npy', obj_cloud)

        #-- gt info 
        index = next(i for i, s in enumerate(bbox_mapping_list) if mesh_name in s)
        bbox_info = bbox_data[index]
        x1, y1, x2, y2, visib_fract = bbox_info[1], bbox_info[2], bbox_info[3], bbox_info[4], 1 - bbox_info[5]
        bbox_obj = np.array([x1, y1, x2-x1, y2-y1], dtype=np.float32)         # x y w h
        out_gtinfo_list.append({"bbox_obj": bbox_obj.tolist(), "px_count_visib": int(px_count_visib), "visib_fract": float(visib_fract)})
        # cv2.imwrite(f"results_render_test/debug_mask/{px_count_visib}_{visib_fract}_{valid_obj_index}.png", load_mask * 255)

        #-- gt
        ob_in_world = np.array(state_list["objects"][mesh_name]['transform_matrix_world']).reshape(4, 4).T
        ob_in_world, scale = normalizeRotation(ob_in_world)
        ob_in_cam = world_in_cam @ ob_in_world
        cam_R_m2c = ob_in_cam[:3, :3].flatten().tolist()
        cam_t_m2c = (ob_in_cam[:3, 3] * 1000).flatten().tolist()    # convert "m" to "mm"
        out_gt_list.append({"cam_R_m2c": cam_R_m2c, "cam_t_m2c": cam_t_m2c, "obj_id": int(obj_id)})

        valid_obj_index += 1

    # # -- save gt_info.json
    with open(os.path.join(scene_dir, f'00000{scene_id}.gt_info.json'), 'w') as f:
        f.write("[")
        for idx, item in enumerate(out_gtinfo_list):
            json.dump(item, f, ensure_ascii=False)
            if idx < len(out_gtinfo_list) - 1:
                f.write(",\n")
        f.write("]")

    # # -- save gt.json
    with open(os.path.join(scene_dir, f'00000{scene_id}.gt.json'), 'w', encoding="utf-8") as f:
        f.write("[")
        for idx, item in enumerate(out_gt_list):
            json.dump(item, f, ensure_ascii=False)
            if idx < len(out_gt_list) - 1:
                f.write(",\n")
        f.write("]")

    # # -- save mask_visib.json
    with open(os.path.join(scene_dir, f'00000{scene_id}.mask_visib.json'), 'w') as f:
        json.dump(out_mask_visib_list, f, indent=2, separators=(",", ":"))


def process_scene(scene_surb_dir, scene_id, data_root):
    try:
        scene_dir = data_root + scene_surb_dir
        process_one_scene(scene_dir, scene_id)
    except Exception as e:
        print(f"Error processing {scene_surb_dir} with scene_id {scene_id}: {e}")

def update_progress(result):
    pbar.update()

def run_parallel_processing(scene_list, data_root):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:  # 限制并发进程数量
        global pbar
        pbar = tqdm(total=len(scene_list) * 2)
        for scene_surb_dir in scene_list:
            pool.apply_async(process_scene, args=(scene_surb_dir, 0, data_root), callback=update_progress)
        for scene_surb_dir in scene_list:
            pool.apply_async(process_scene, args=(scene_surb_dir, 1, data_root), callback=update_progress)
        pool.close()
        pool.join()
        pbar.close()

if __name__ == "__main__" :

    # init path
    data_root = 'data/Training-Data/FoundationPose-Objaverse/train_pbr_web/'
    scene_list_dir = 'data/Training-Data/FoundationPose-Objaverse/scene_list.json'   # ["5261509/1007153753", "5261509/1013578302", ...]
    with open(scene_list_dir, 'r') as f:
        scene_list = json.load(f)

    run_parallel_processing(scene_list, data_root)
    for scene_surb_dir in tqdm(scene_list):
        scene_dir = data_root + scene_surb_dir
        scene_id = 0
        process_one_scene(scene_dir, scene_id)
    for scene_surb_dir in tqdm(scene_list):
        scene_dir = data_root + scene_surb_dir
        scene_id = 1
        process_one_scene(scene_dir, scene_id)
    print("over!")
    