import os
import shutil

base_dir = 'sweep_000_lr=1e-5_opt=lion_embedsize=2048'

lrs = [
    '1e-5',
    '1e-6',
    '1e-3',
    '1e-4'
]

opts = [
    'lion',
    'radam'
]

# encacts = [
#     'MAX',
#     'MEAN'
# ]

embedsizes = [
    '2048',
    '4096',
    '1024'
]

count = 0

for lr in lrs:
    for opt in opts:
        for embedsize in embedsizes:
            if lr == '1e-5' and opt == 'lion' and embedsize == '2048': continue
            count += 1
            dest_name = f"sweep_{count:03}_lr={lr}_opt={opt}_embedsize={embedsize}"
            shutil.copytree(base_dir, dest_name)
            print(f'created {dest_name}')