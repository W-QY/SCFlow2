import csv
import argparse
import os
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', default='results/ycbv/ponintnet2+sam6d')
    parser.add_argument('bop19_test_json', default='data/ycbv/test_targets_bop19.json')
    parser.add_argument('save_path', default='results/bop_results/dgflow+flow_ycbv-test.csv')
    parser.add_argument('--base_time', default=0, type=float)
    args = parser.parse_args()
    return args

def map_R_to_str(rotation):
    str_rotation = []
    for i, ele in enumerate(rotation):
        if i == 8:
            str_rotation.append(str(ele))
            continue
        str_rotation.append(str(ele)+' ')
    return ''.join(str_rotation)

def map_T_to_str(translation):
    str_translation = []
    for i, ele in enumerate(translation):
        if i == 2:
            str_translation.append(str(ele))
            continue
        str_translation.append(str(ele)+' ')
    return ''.join(str_translation)
    




if __name__ == '__main__':
    header = ['scene_id', 'im_id','obj_id','score','R','t','time']
    args = parse_args()
    result_dir, bop19_test_json, save_path = args.result_dir, args.bop19_test_json, args.save_path
    base_time = args.base_time
    scenes = os.listdir(result_dir)
    with open(bop19_test_json, 'r') as f:
        bop19_test_targets = json.load(f)
    bop19_test_scenes = [t['scene_id'] for t in bop19_test_targets]
    bop19_test_scenes = sorted(list(set(bop19_test_scenes)))
    bop19_test_scenes_images = {s:[] for s in bop19_test_scenes}
    for t in bop19_test_targets:
        scene_id, image_id = int(t['scene_id']), int(t['im_id'])
        if image_id not in bop19_test_scenes_images[scene_id]:
            bop19_test_scenes_images[scene_id].append(image_id)
        
    
    save_content = []
    for scene_id in bop19_test_scenes_images:
        scene_result_json = os.path.join(result_dir, f"{scene_id:06d}", "scene_gt.json")
        with open(scene_result_json, 'r') as f:
            scene_results = json.load(f)
        scene_image_ids = bop19_test_scenes_images[scene_id]
        for image_id in scene_image_ids:
            if str(image_id) not in scene_results:
                continue
            image_preds = scene_results[str(image_id)]
            for pred_objs in image_preds:
                pred_r = np.array(pred_objs['cam_R_m2c']).reshape(9).tolist()
                pred_t = np.array(pred_objs['cam_t_m2c']).reshape(3).tolist()
                obj_id = pred_objs['obj_id']
                time = pred_objs.get('time', -1.0) + base_time
                # time = base_time
                save_content.append(
                    [scene_id, image_id, obj_id, 1., map_R_to_str(pred_r), map_T_to_str(pred_t), time]
                )
    with open(save_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(save_content)