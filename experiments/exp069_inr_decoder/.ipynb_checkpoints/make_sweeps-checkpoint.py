import os
import shutil

base_dir = 'sweep_000_lr=1e-5_opt=lion_fixedtrainingset=T_fixedrandi=T_layersize=128_numlayers=5'

lrs = [
    '1e-5',
    '1e-6',
    # '1e-3',
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

# embedsizes = [
#     '2048',
#     '4096',
#     '1024'
# ]

fixedtrainingsets = [
    'T',
    'F'
]

fixedrandis = [
    'T',
    'F'
]

layersizes = [
    '128',
]

numlayerses = [
    '5',
]

count = 0
for fixedtrainingset in fixedtrainingsets:
    for fixedrandi in fixedrandis:
        for lr in lrs:
            for opt in opts:
                for layersize in layersizes:
                    for numlayers in numlayerses:
                        if f'sweep_000_lr={lr}_opt={opt}_fixedtrainingset={fixedtrainingset}_fixedrandi={fixedrandi}_layersize={layersize}_numlayers={numlayers}' == base_dir: continue
                        count += 1
                        dest_name = f'sweep_{count:03}_lr={lr}_opt={opt}_fixedtrainingset={fixedtrainingset}_fixedrandi={fixedrandi}_layersize={layersize}_numlayers={numlayers}'
                        shutil.copytree(base_dir, dest_name)
                        print(f'created {dest_name}')