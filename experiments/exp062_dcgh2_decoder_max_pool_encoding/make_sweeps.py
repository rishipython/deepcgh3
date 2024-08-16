import os
import shutil

base_dir = 'sweep_000_lr=1e-3_opt=radam_encact=MAX'

lrs = [
    '1e-3',
    '1e-4',
    '1e-5',
    '1e-6'
]

opts = [
    'radam',
    'lion'
]

encacts = [
    'MAX',
    'MEAN'
]

count = 0

for lr in lrs:
    for opt in opts:
        for encact in encacts:
            if lr == '1e-3' and opt == 'radam' and encact == 'MAX': continue
            count += 1
            dest_name = f"sweep_{count:03}_lr={lr}_opt={opt}_encact={encact}"
            shutil.copytree(base_dir, dest_name)
            print(f'created {dest_name}')