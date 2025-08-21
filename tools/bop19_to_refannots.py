import csv, os, argparse, json
from pathlib import Path
# from datasets.utils import dumps_json

dataset_name = 'ycbv'    # ['lmo', 'tless', 'tudl', 'icbin', 'itodd', 'hb', 'ycbv']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default=f'data/initial_poses/initpose_csv/foundpose/foundpose_{dataset_name}-test.csv')
    parser.add_argument('--save_dir', default=f'data/initial_poses/foundpose/{dataset_name}')
    args = parser.parse_args()
    return args


def dumps_json(data, indent=2, depth=2):
    assert depth > 0
    space = ' '*indent
    s = json.dumps(data, indent=indent)
    lines = s.splitlines()
    N = len(lines)
    # determine which lines to be shortened
    is_over_depth_line = lambda i: i in range(N) and lines[i].startswith(space*(depth+1))
    is_open_bracket_line = lambda i: not is_over_depth_line(i) and is_over_depth_line(i+1)
    is_close_bracket_line = lambda i: not is_over_depth_line(i) and is_over_depth_line(i-1)
    # 
    def shorten_line(line_index):
        if not is_open_bracket_line(line_index):
            return lines[line_index]
        # shorten over-depth lines
        start = line_index
        end = start
        while not is_close_bracket_line(end):
            end += 1
        has_trailing_comma = lines[end][-1] == ','
        _lines = [lines[start][-1], *lines[start+1:end], lines[end].replace(',','')]
        d = json.dumps(json.loads(' '.join(_lines)))
        return lines[line_index][:-1] + d + (',' if has_trailing_comma else '')
    # 
    s = '\n'.join([
        shorten_line(i)
        for i in range(N) if not is_over_depth_line(i) and not is_close_bracket_line(i)
    ])
    #
    return s


def decode_R(rotation:str):
    rotation = rotation.split(' ')
    assert len(rotation) == 9
    rotation = [float(ele) for ele in rotation]
    return rotation

def decode_t(translation:str):
    translation = translation.split(' ')
    assert len(translation) == 3
    translation = [float(ele) for ele in translation]   
    return translation


if __name__ == '__main__':
    args = parse_args()
    csv_path, save_dir = args.csv_path, args.save_dir
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        reference_poses = []
        for i, line  in enumerate(reader):
            if i == 0:
                # header
                pass
            else:
                reference_poses.append(line[0].split(','))
    scene_poses = dict()
    for pose in reference_poses:
        scene_id, img_id, obj_id = pose[0], pose[1], pose[2]
        score, r, t = pose[3], pose[4], pose[5]
        rotation = decode_R(r)
        translation = decode_t(t)
        if scene_id not in scene_poses:
            scene_poses[scene_id] = dict()
        if img_id not in scene_poses[scene_id]:
            scene_poses[scene_id][img_id] = []
        scene_poses[scene_id][img_id].append(
            dict(
                cam_t_m2c = translation,
                cam_R_m2c = rotation,
                obj_id = int(obj_id),
                score= float(score),
            )
        )
    # sort image ids
    for scene_id in scene_poses:
        keys = list(scene_poses[scene_id].keys())
        keys = sorted([int(k) for k in keys])
        keys = [str(k) for k in keys]
        scene_poses[scene_id] = {k:scene_poses[scene_id][k] for k in keys}
    for scene_id in scene_poses:
        sequence_dir = Path(os.path.join(save_dir, f"{int(scene_id):06d}"))
        sequence_dir.mkdir(exist_ok=True, parents=True)
        save_path = sequence_dir.joinpath('scene_gt.json')
        poses = scene_poses[scene_id]
        save_content = dumps_json(poses)
        save_path.write_text(save_content)
        