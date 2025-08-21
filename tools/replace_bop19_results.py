import csv, os, argparse, json
from pathlib import Path
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='results/bop_results/dgflow_ycbv-test.csv')
    parser.add_argument('--json_path', default='data/ycbv/test_bop19')
    parser.add_argument('--save_path', default='results/bop_results/dgflow+foundpose+gt_ycbv-test.csv')
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
    args = parse_args()
    csv_path, json_path, save_path = args.csv_path, args.json_path, args.save_path
    base_time = 0
    header = ['scene_id', 'im_id','obj_id','score','R','t','time']
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        reference_poses = []
        for i, line  in enumerate(reader):
            if i == 0:
                # header
                pass
            else:
                reference_poses.append(line[0].split(','))
    # reference_poses_dict = {tuple(item[:3]): item for item in reference_poses}

    # update_preds = dict()
    json_results = dict()
    for subdir in os.listdir(json_path):
        subdir_path = os.path.join(json_path, subdir)
        json_file_path = os.path.join(subdir_path, 'scene_gt.json')
        with open(json_file_path, 'r') as f:
            datas = json.load(f)
        # update_preds.append()
        scene_id = str(int(subdir))
        for image_id in datas:
            pred_results = datas[str(image_id)]
            for pred_objs in pred_results:
                pred_r = np.array(pred_objs['cam_R_m2c']).reshape(9).tolist()
                pred_t = np.array(pred_objs['cam_t_m2c']).reshape(3).tolist()
                obj_id = str(pred_objs['obj_id'])
                time = str(-1.0) # pred_objs.get('time', -1.0) + base_time
                if scene_id not in json_results:
                    json_results[scene_id] = dict()
                if image_id not in json_results[scene_id]:
                    json_results[scene_id][image_id] = dict()
                if obj_id not in json_results[scene_id][image_id]:
                    json_results[scene_id][image_id][obj_id] = []
                json_results[scene_id][image_id][obj_id].append(
                        [scene_id, image_id, obj_id, '1.0', map_R_to_str(pred_r), map_T_to_str(pred_t), time])
                    
    data_len = len(reference_poses)
    for i in tqdm(range(data_len)):
        pose = reference_poses[i]
        scene_id, image_id, obj_id = pose[0], pose[1], pose[2]
        if scene_id in json_results:
            if image_id in json_results[scene_id]:
                if obj_id in json_results[scene_id][image_id]:
                    reference_poses[i] = json_results[scene_id][image_id][obj_id][0] # .pop(0)

            # result_key = tuple(results[:3])
            # reference_poses_dict[result_key] = results
    # reference_poses.sort()
    # reference_poses = list(reference_poses_dict.values())

    with open(save_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(reference_poses)
            


        
    