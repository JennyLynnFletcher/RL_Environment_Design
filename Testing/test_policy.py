import gym, ray
from ray.rllib.agents import ppo
import json
import pickle

import sys
sys.path.append('/home/jenny/Documents/Part II Project/Code/')


from env import ObstaclesEnv



ray.init()        
        
trainer = ppo.PPOTrainer(env=ObstaclesEnv, config={
    "framework": "torch",
    "num_workers": 12,
    "num_gpus": 1,
    "use_critic": True,
    "use_gae": True,
    "lambda": 0.95,
    "kl_coeff": 0.5,
    "rollout_fragment_length": 200,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "shuffle_sequences": True,
    "num_sgd_iter": 30,
    "lr": 5e-5, #1e-4
    "lr_schedule": None,
    "vf_loss_coeff": 1.0,
    "model": {
        "dim": 12,
        "conv_filters": [
        [25, [2, 2], 2],
        [25, [2, 2], 2],
        [25, [2, 2], 2],
        [25, [2, 2], 2],],
        "vf_share_layers": False,
    },
    "entropy_coeff": 0.01,
    "entropy_coeff_schedule": None,
    "clip_param": 0.1,
    "vf_clip_param": 10.0,
    "grad_clip": None,
    "kl_target": 0.01,
    "batch_mode": "truncate_episodes",
    "observation_filter": "NoFilter",
    "simple_optimizer": False,
    "_fake_gpus": False,})
        
    
cfg = {'env_config': {}, 'model': {}}
with open("/home/jenny/ray_results/PPO_ObstaclesEnv_2021-05-09_05-47-02g8k84ted/params.json") as json_file:
    cfg = json.load(json_file)
    
print(cfg)
trial = 0
trainer.restore(str("/home/jenny/ray_results/PPO_ObstaclesEnv_2021-05-09_05-47-02g8k84ted/checkpoint_147/checkpoint-147"))
env = ObstaclesEnv(cfg)
env.seed(1)
obs = env.reset()


results = []
done = False
for j in range(500):
    i = 0
    while not done:
        actions = trainer.compute_action(obs)
        obs, reward, done, info = env.step(actions)
        results.append({
                'instance': j,
                'step': i,
                'trial': trial,
                'reward': reward,
                'percentage_dist': info['percentage_dist'],
                'percentage_speed': info['percentage_speed']
            })
        i += 1
        if done:
            env.render(save=True)
        #else:
            #env.render()
    print(results)
    obs = env.reset()
    done = False


f = open('/home/jenny/ray_results/PPO_ObstaclesEnv_2021-05-09_05-47-02g8k84ted/checkpoint_147/results.pkl', 'wb')
pickle.dump(results, f)
f.close()
