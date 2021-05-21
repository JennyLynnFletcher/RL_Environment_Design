from colour import Color
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'serif': 'cm'})
rc('text', usetex=True)

gradient_0 = list(Color("green").range_to(Color("blue"),500))

fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_aspect(1)

for i in range(500):
    plt.bar(i,10,width=1, color=gradient_0[i].hex)

plt.xlabel("Timesteps")

frame1 = plt.gca()
frame1.axes.get_yaxis().set_visible(False)
fig.tight_layout()

plt.savefig("grad_key.pdf")
