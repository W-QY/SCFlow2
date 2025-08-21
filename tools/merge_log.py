import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', type=str, nargs='+', help='json logs to be merged')
    parser.add_argument('--time_stamps', type=int, nargs='+', help='time stamps for specific each log')
    parser.add_argument('--out', type=str, help='save the merged log')
    args = parser.parse_args()
    return args 

def find_index(logs_json, time_stamp):
    for i, log in enumerate(logs_json):
        if i == 0:
            continue
        iter = json.loads(log).pop('iter')
        if iter == time_stamp:
            return i

if __name__ == '__main__':
    args = parse_args()
    logs_json, time_stamps = args.logs, args.time_stamps 
    assert len(logs_json) == len(time_stamps) + 1
    new_logs = []
    for i, log_json in enumerate(logs_json):
        with open(log_json, 'r') as f:
            log_json = f.readlines()
        if i == 0:
            new_logs.append(log_json[0])
        
        if i != len(logs_json) - 1:
            index = find_index(log_json, time_stamps[i])
            new_logs.extend(log_json[1:index])
        else:
            new_logs.extend(log_json[1:])
    with open(args.out, 'w') as f:
        f.writelines(new_logs)
