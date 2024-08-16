import os
import shutil

base_dir = 'sweep_000_lr=1e-4_opt=radam_ff=5'

lrs = [
    '1e-4',
    '1e-3',
    '1e-5',
    '1e-6'
]

opts = [
    'radam',
    'lion'
]

ffs = [
    '5',
    '10',
    '15'
]

count = 0

for lr in lrs:
    for opt in opts:
        for ff in ffs:
            if lr == '1e-4' and opt == 'radam' and ff == '5': continue
            count += 1
            dest_name = f"sweep_{count:03}_lr={lr}_opt={opt}_ff={ff}"
            shutil.copytree(base_dir, dest_name)
            print(f'created {dest_name}')