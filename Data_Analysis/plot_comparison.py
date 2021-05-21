import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'serif': 'cm'})
rc('text', usetex=True)

results_147 = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-05-09_05-47-02g8k84ted/checkpoint_147/results.pkl', "rb" ) )
results_1 = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-06_17-29-13embh0dvs/checkpoint_1/results.pkl', "rb" ) )


instances_147 = []
i = 0
instance_147 = []
for result in results_147:
    if result['instance'] > i:
        instances_147.append(instance_147)
        instance_147 = []
        i += 1
    else:
        instance_147.append(result)


rewards_147 = []
early_termination_147 = 0
rewards_cumulative_147 = []
for instance in instances_147:
    reward_147 = []
    reward_147_cumulative = 0
    for i in instance:
        r = i['reward']
        reward_147_cumulative += r
        if r > -100:
            pass
        else:
            early_termination_147 += 1
        reward_147.append(r)
    rewards_147.append(reward_147)
    rewards_cumulative_147.append(reward_147_cumulative)

##################################################################################################    
    
instances_1 = []
i = 0
instance_1 = []
for result in results_1:
    if result['instance'] > i:
        instances_1.append(instance_1)
        instance_1 = []
        i += 1
    else:
        instance_1.append(result)


rewards_1 = []
early_termination_1 = 0
rewards_cumulative_1 = []
for instance in instances_1:
    reward_1 = []
    reward_1_cumulative = 0
    for i in instance:
        r = i['reward']
        reward_1_cumulative += r
        if r > -100:
            pass
        else:
            early_termination_1 += 1
        reward_1.append(r)
    rewards_1.append(reward_1)
    rewards_cumulative_1.append(reward_1_cumulative)
    
##################################################################################################

end_rewards_147 = []

for i in range(len(rewards_147)): 
    if len(rewards_147[i]) > 0:
        end_rewards_147.append(rewards_147[i][-1])
        
end_rewards_1 = []

for i in range(len(rewards_1)): 
    if len(rewards_1[i]) > 0:
        end_rewards_1.append(rewards_1[i][-1])

###################################################################################################

data = np.array([end_rewards_1,end_rewards_147])

fig, ax = plt.subplots()
ax.set_aspect(0.035)
labels = ["Checkpoint 1", "Checkpoint 147"]
ax.set_xticks([1,4])
ax.set_xticklabels(labels)

#plt.violinplot(data, [1,4], showmeans=False, showextrema=True, showmedians=True, widths=3.5)
plt.ylabel("Reward")


#Begin section from MatPlotLib docs
parts = ax.violinplot(data, [1,4], showmeans=False, showextrema=False, showmedians=False, widths=3.5)

for pc in parts['bodies']: 
    pc.set_facecolor('#1f77b4')
    pc.set_edgecolor('black')
    pc.set_alpha(1)
    
quartile1 = []
medians = []
quartile3 =[]

q1, m, q3 = np.percentile(end_rewards_1, [25, 50, 75])
quartile1.append(q1)
medians.append(m)
quartile3.append(q3)
q1, m, q3 = np.percentile(end_rewards_147, [25, 50, 75])
quartile1.append(q1)
medians.append(m)
quartile3.append(q3)

inds = [1,4]
ax.scatter(inds, medians, marker='o', facecolors='white', edgecolors='black', s=15, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)

plt.savefig("/home/jenny/Documents/Part II Project/Code/graphs/rewards_end_comparison.pdf", bbox_inches='tight', pad_inches=0)

#End section from MatPlotLib docs

plt.clf()
plt.hist(data, bins=50, stacked=True)
plt.xlabel("Reward")
plt.savefig("/home/jenny/Documents/Part II Project/Code/graphs/rewards_hist_1.pdf")

plt.clf()
plt.hist(end_rewards_147, bins=50)
plt.xlabel("Reward")
plt.savefig("/home/jenny/Documents/Part II Project/Code/graphs/rewards_hist_147.pdf")

print(early_termination_1/499)
print(early_termination_147/499)















