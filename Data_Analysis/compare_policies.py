import pickle
import math

#results = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-15_18-53-36kiwk8n0o/checkpoint_720/results.pkl', "rb" ) )
#results = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-15_18-53-36kiwk8n0o/checkpoint_649/results.pkl', "rb" ) )
results_440 = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-12_07-28-31nsbm6pce/checkpoint_440/results.pkl', "rb" ) )
#results = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-09_07-00-579qrzfh07/checkpoint_392/results.pkl', "rb" ) )
#results = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-09_07-00-579qrzfh07/checkpoint_310/results.pkl', "rb" ) )
#results = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-06_17-29-13embh0dvs/checkpoint_192/results.pkl', "rb" ) )
results_1 = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-06_17-29-13embh0dvs/checkpoint_1/results.pkl', "rb" ) )


instances = []
i = 0
instance = []
for result in results_440:
    if result['instance'] > i:
        instances.append(instance)
        instance = []
        i += 1
    else:
        instance.append(result)

rewards = []
early_termination_440 = 0

for instance in instances:
    reward = []
    for i in instance:
        r = i['reward']
        if r > -100:
            reward.append(r)
            early_termination_440 += 1
            
    rewards.append(reward)

##########################################################################

instances = []
i = 0
instance = []
for result in results_1:
    if result['instance'] > i:
        instances.append(instance)
        instance = []
        i += 1
    else:
        instance.append(result)

rewards = []
early_termination_1 = 0

for instance in instances:
    reward = []
    for i in instance:
        r = i['reward']
        if r > -100:
            reward.append(r)
            early_termination_1 += 1
            
    rewards.append(reward)
    
print(early_termination_440)
print(early_termination_1)
    
