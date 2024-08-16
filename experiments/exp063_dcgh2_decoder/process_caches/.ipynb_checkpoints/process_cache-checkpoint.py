import pickle
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
print(jax.devices())

exp = 'exp063_dcgh2_decoder/sweep_000_lr=1e-5_opt=lion_embedsize=2048'
exp_sweep = exp.split('/')[1][:9]
os.makedirs(exp_sweep, exist_ok = True) 
epoch = 44000
cache = pickle.load(open(f'/nrs/turaga/athavaler/experiments/{exp}/cache_epoch_{epoch}.pkl', 'rb'))
for i in range(100):
    print(i)
    cache_test = cache[f'cache_test_{i}']
    pickle.dump(cache_test, open(f'{exp_sweep}/cache_test_{i}.pkl', 'wb'))