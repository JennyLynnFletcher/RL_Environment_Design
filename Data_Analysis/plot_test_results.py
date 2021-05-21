import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'serif': 'cm'})
rc('text', usetex=True)


#results = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-15_18-53-36kiwk8n0o/checkpoint_720/results.pkl', "rb" ) )
#results = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-15_18-53-36kiwk8n0o/checkpoint_649/results.pkl', "rb" ) )
#results = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-12_07-28-31nsbm6pce/checkpoint_440/results.pkl', "rb" ) )
#results = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-09_07-00-579qrzfh07/checkpoint_392/results.pkl', "rb" ) )
#results = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-09_07-00-579qrzfh07/checkpoint_310/results.pkl', "rb" ) )
#results = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-06_17-29-13embh0dvs/checkpoint_192/results.pkl', "rb" ) )
#results = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-04-06_17-29-13embh0dvs/checkpoint_1/results.pkl', "rb" ) )

#results = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-05-07_02-30-36rtvpdb8n/checkpoint_82/results.pkl', "rb" ) )

results = pickle.load( open( '/home/jenny/ray_results/PPO_ObstaclesEnv_2021-05-09_05-47-02g8k84ted/checkpoint_147/results.pkl', "rb" ) )

checkpoint = "147"

instances = []
i = 0
instance = []
for result in results:
    if result['instance'] > i:
        instances.append(instance)
        instance = []
        i += 1
    else:
        instance.append(result)

plt.clf()


rewards = []
reward_cumulative = 0
for instance in instances:
    reward = []
    for i in instance:
        r = i['reward']
        reward_cumulative += r
        if r > -100:
            reward.append(r)
    rewards.append(reward)

print("Episode Reward Average: ", reward_cumulative/len(instances))
    
dists = []
for instance in instances:
    dist = []
    for i in instance:
        d = i['percentage_dist']
        if len(d) == 2:
            dist.append(100*(((d[0] + d[1])/2) - 1))
    dists.append(dist)

    
speeds = []
for instance in instances:
    speed = []
    for i in instance:
        s = i['percentage_speed']
        if len(s) == 2:
            speed.append(100*(s[0] + s[1])/2)
    speeds.append(speed)
    
for i in range(len(rewards)):     
    plt.plot(list(range(0, len(rewards[i]))),rewards[i],label = 'id %s'%i)
plt.xlabel(r"Steps")
plt.ylabel(r"Reward")
plt.title(r"Checkpoint {}".format(checkpoint))
plt.savefig("/home/jenny/Documents/Part II Project/Code/graphs/rewards_{}.pdf".format(checkpoint))

plt.clf()    
plt.ylim(-10, 120)
for i in range(len(dists)):     
    plt.plot(list(range(0, len(dists[i]))),dists[i],label = 'id %s'%i)
plt.xlabel(r"Steps")
plt.ylabel(r"Detour Percentage")
plt.title(r"Checkpoint {}".format(checkpoint))
plt.savefig("/home/jenny/Documents/Part II Project/Code/graphs/dists_{}.pdf".format(checkpoint))

plt.clf()    

for i in range(len(speeds)):     
    plt.plot(list(range(0, len(speeds[i]))),speeds[i],label = 'id %s'%i)
plt.xlabel(r"Steps")
plt.ylabel(r"average % speed of preferred speed")
plt.title(r"Checkpoint {}".format(checkpoint))
plt.savefig("/home/jenny/Documents/Part II Project/Code/graphs/speeds_{}.pdf".format(checkpoint))

plt.clf()

#for i in range(len(rewards)):     
    #if len(rewards[i]) > 0:
        #plt.plot([0, len(rewards[i])],[rewards[i][0], rewards[i][-1]],label = 'id %s'%i)
#plt.xlabel("xlabel")
#plt.ylabel("ylabel")
#plt.title("Ckpt {}".format(checkpoint))
#plt.savefig("/home/jenny/Documents/Part II Project/PyBullet/graphs/rewards_end_{}.png".format(checkpoint))

#plt.clf()

#for i in range(len(dists)): 
    #if len(dists[i]) > 0:
        #plt.plot([0, len(dists[i])],[dists[i][0], dists[i][-1]],label = 'id %s'%i)
#plt.xlabel("xlabel")
#plt.ylabel("ylabel")
#plt.title("title")
#plt.savefig("/home/jenny/Documents/Part II Project/PyBullet/graphs/dists_end_{}.png".format(checkpoint))

#plt.clf()

#for i in range(len(speeds)): 
    #if len(speeds[i]) > 0:
        #plt.plot([0, len(speeds[i])],[speeds[i][0], speeds[i][-1]],label = 'id %s'%i)
#plt.xlabel("xlabel")
#plt.ylabel("ylabel")
#plt.title("title")
#plt.savefig("/home/jenny/Documents/Part II Project/PyBullet/graphs/speeds_end_{}.png".format(checkpoint))

#plt.clf()

labels = ["begin", "end"]

begin_dists = []
end_dists = []

dist_intervals = [[],[],[],[],[],[],[]]

for i in range(len(dists)): 
    if len(dists[i]) > 0:
        begin_dists.append(dists[i][0])
        end_dists.append(dists[i][-1])
        for j in range(7):
            if len(dists[i]) > j*4:
                dist_intervals[j].append(dists[i][j*4])
            else:
                dist_intervals[j].append(dists[i][-1])

#plt.boxplot([begin_dists, end_dists], labels=labels)
#plt.savefig("/home/jenny/Documents/Part II Project/PyBullet/graphs/dists_box_{}.png".format(checkpoint))

plt.clf()
plt.ylim(-10, 140)
labels = [i for i in range(0,28,4)]
bp = plt.boxplot(dist_intervals, labels=labels, notch=True, showfliers=False)
plt.setp(bp['medians'], color='#1f77b4')
plt.xlabel(r"Steps")
plt.ylabel(r"Detour Percentage")
plt.title(r"Checkpoint {}".format(checkpoint))
plt.savefig("/home/jenny/Documents/Part II Project/Code/graphs/dists_box_prog_{}.pdf".format(checkpoint))

#plt.clf()
#plt.violinplot([begin_dists, end_dists], [0, 23], showmeans=False, showextrema=True, showmedians=True, widths=3.5)
#plt.xlabel("xlabel")
#plt.ylabel("ylabel")
#plt.title("Ckpt {}".format(checkpoint))
#plt.savefig("/home/jenny/Documents/Part II Project/PyBullet/graphs/dists_violin_{}.png".format(checkpoint))

plt.clf()
plt.ylim(90, 220)
plt.violinplot(dist_intervals, labels, showmeans=False, showextrema=True, showmedians=True, widths=3.5)
plt.xlabel(r"Steps")
plt.ylabel(r"Detour Percentage")
plt.title(r"Checkpoint {}".format(checkpoint))
plt.savefig("/home/jenny/Documents/Part II Project/Code/graphs/dists_violin_prog_{}.pdf".format(checkpoint))

plt.clf()

labels = ["begin", "end"]

begin_rewards = []
end_rewards = []

reward_intervals = [[],[],[],[],[],[],[]]

for i in range(len(rewards)): 
    if len(rewards[i]) > 0:
        begin_rewards.append(rewards[i][0])
        end_rewards.append(rewards[i][-1])
        for j in range(7):
            if len(rewards[i]) > j*4:
                reward_intervals[j].append(rewards[i][j*4])
            else:
                reward_intervals[j].append(rewards[i][-1])
            

#plt.boxplot([begin_rewards, end_rewards], labels=labels, showfliers=False)
#plt.xlabel("xlabel")
#plt.ylabel("ylabel")
#plt.title("Ckpt {}".format(checkpoint))
#plt.savefig("/home/jenny/Documents/Part II Project/PyBullet/graphs/rewards_box_{}.png".format(checkpoint))

plt.clf()
plt.ylim(4.7, 10.5)
labels = [i for i in range(0,28,4)]
plt.boxplot(reward_intervals, labels=labels, notch=True, showfliers=False)
plt.xlabel(r"Steps")
plt.ylabel(r"Reward")
plt.title(r"Checkpoint {}".format(checkpoint))
plt.savefig("/home/jenny/Documents/Part II Project/Code/graphs/rewards_box_prog_{}.pdf".format(checkpoint))

#plt.clf()
#plt.violinplot([begin_rewards, end_rewards], [0, 23], showmeans=False, showextrema=True, showmedians=True, widths=3.5)
#plt.xlabel("xlabel")
#plt.ylabel("ylabel")
#plt.title("Ckpt {}".format(checkpoint))
#plt.savefig("/home/jenny/Documents/Part II Project/PyBullet/graphs/rewards_violin_{}.png".format(checkpoint))

plt.clf()
plt.violinplot(reward_intervals, labels, showmeans=False, showextrema=False, showmedians=True, widths=3.5)
plt.xlabel(r"Steps")
plt.ylabel(r"Reward")
plt.title(r"Checkpoint {}".format(checkpoint))
plt.savefig("/home/jenny/Documents/Part II Project/Code/graphs/rewards_violin_prog_{}.pdf".format(checkpoint))

plt.clf()

labels = ["begin", "end"]

begin_speeds = []
end_speeds = []

speed_intervals = [[],[],[],[],[],[],[]]

for i in range(len(speeds)): 
    if len(speeds[i]) > 0:
        begin_speeds.append(speeds[i][0])
        end_speeds.append(speeds[i][-1])
        for j in range(7):
            if len(speeds[i]) > j*4:
                speed_intervals[j].append(speeds[i][j*4])
            else:
                speed_intervals[j].append(speeds[i][-1])
                
            

#plt.boxplot([begin_speeds, end_speeds], labels=labels, showfliers=False)
#plt.xlabel("xlabel")
#plt.ylabel("ylabel")
#plt.title("Ckpt {}".format(checkpoint))
#plt.savefig("/home/jenny/Documents/Part II Project/PyBullet/graphs/speeds_box_{}.png".format(checkpoint))

plt.clf()
labels = [i for i in range(0,28,4)]
plt.boxplot(speed_intervals, labels=labels, notch=True, showfliers=False, whis=[10, 90])
plt.xlabel(r"Steps")
plt.ylabel("average percentage speed of preferred speed")
plt.title(r"Checkpoint {}".format(checkpoint))
plt.savefig("/home/jenny/Documents/Part II Project/Code/graphs/speeds_box_prog_{}.pdf".format(checkpoint))

#plt.clf()
#plt.violinplot([begin_speeds, end_speeds], [0, 23], showmeans=False, showextrema=True, showmedians=True, widths=3.5)
#plt.xlabel("xlabel")
#plt.ylabel("ylabel")
#plt.title("Ckpt {}".format(checkpoint))
#plt.savefig("/home/jenny/Documents/Part II Project/PyBullet/graphs/speeds_violin_{}.png".format(checkpoint))

plt.clf()

fig, ax = plt.subplots()

#Begin section from MatPlotLib docs
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

parts = ax.violinplot(speed_intervals, labels, showmeans=False, showextrema=False, showmedians=False, widths=3.5)

for pc in parts['bodies']:
    pc.set_facecolor('#1f77b4')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(speed_intervals, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(sorted(speed_intervals), quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = labels
ax.scatter(inds, medians, marker='o', facecolors='white', edgecolors='white', s=30, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
#End section from MatPlotLib docs

plt.xlabel(r"Steps")
plt.ylabel("average percentage speed of preferred speed")
plt.title(r"Checkpoint {}".format(checkpoint))
plt.savefig("/home/jenny/Documents/Part II Project/Code/graphs/speeds_violin_prog_{}.pdf".format(checkpoint))



