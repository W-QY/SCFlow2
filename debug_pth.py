import torch

checkpoint_dir = 'scflow2_pretrained.pth'
checkpoint = torch.load(checkpoint_dir)

if 'meta' in checkpoint:
    del checkpoint['meta']
if 'optimizer' in checkpoint:
    del checkpoint['optimizer']

# 获取 state_dict
state_dict = checkpoint['state_dict']

# 找到需要删除的 key
keys_to_delete = [k for k in state_dict.keys() if k.startswith("decoder.point_flow_pose_pred")]

# 逐个删除
for k in keys_to_delete:
    del state_dict[k]

torch.save(checkpoint, "scflow2_pretrained.pth")

print("debug!")