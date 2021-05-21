import gym, ray
from ray.rllib.agents import ppo

from env import ObstaclesEnv

ray.shutdown()    
ray.init(_temp_dir="/home/jlf60/tmp")        
        
trainer = ppo.PPOTrainer(env=ObstaclesEnv, config={
    "framework": "torch",
    "num_workers": 64,
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
    "lr": 1e-4, #1e-5
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


########## FOR USE IF TRAINING STARTS FROM LATER CHACKPOINT ##################

#cfg = {'env_config': {}, 'model': {}}
#with open("/home/jlf60/ray_results/PPO_ObstaclesEnv_2021-04-06_17-29-13embh0dvs/checkpoint_1/params.json") as json_file:
    #cfg = json.load(json_file)
    
#print(cfg)
#trial = 0
#trainer.restore(str("/home/jlf60/ray_results/PPO_ObstaclesEnv_2021-04-06_17-29-13embh0dvs/checkpoint_1/checkpoint-1"))
#env = ObstaclesEnv(cfg)
#env.seed(trial)
#obs = env.reset()


for i in range(500):
   # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(result)
    checkpoint = trainer.save()
    print("checkpoint saved at", checkpoint)
