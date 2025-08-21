import argparse
from glob import glob
import os
from os import path as osp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', default='data/tless/test_primesense', type=str)
    parser.add_argument('--save-path', default='data/tless/image_lists/test.txt', type=str)
    parser.add_argument('--pattern', default='*/rgb/*.png', type=str)   #-# .png  // .jpg
    args = parser.parse_args()
    return args




if __name__ =='__main__':
    args = parse_args()
    image_list = glob(osp.join(args.source_dir, args.pattern))
    image_list = sorted(image_list)
    # image_list = [i.replace(args.source_dir, '')+'\n' for i in image_list]   
    image_list = [i.replace(args.source_dir, '').strip('/')+'\n' for i in image_list]
    print(f"Totally {len(image_list)} images found")
    os.makedirs(args.save_path.rsplit('/', 1)[0], exist_ok=True)
    with open(args.save_path, 'w') as f:
        f.writelines(image_list)