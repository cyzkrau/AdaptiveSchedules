import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import gaussian_kde

sigma02 = 0.16
sigmat2 = lambda t: np.exp(-2*t) * (sigma02 - 1) + 1
mu0 = 10
mut = lambda t: np.exp(-t) * mu0

score = lambda x,t: -(x-mut(t))/sigmat2(t)
score_error = lambda t: 10*np.cos(10*t*np.pi) + 10

def run_plan(plan, x_range):
    Yts = np.random.randn(100000)
    for k in range(len(plan)-1,0,-1):
        dt = plan[k] - plan[k-1]
        Yts += (Yts + 2*score(Yts, plan[k])) * dt + np.sqrt(2*dt) * np.random.randn(len(Yts))
        Yts += 2*score_error(plan[k]) * dt
    print(Yts.mean(), Yts.var())
    kde = gaussian_kde(Yts)
    return x_range, kde(x_range)

x_range = np.linspace(0, 20, 100)


cosine_f = lambda t: np.cos((t/1000+0.08)/(1+0.08)*np.pi/2)**2
plan_cosine = -np.log(cosine_f(np.arange(1,1001))/cosine_f(0))/2
plan_uniform = np.linspace(0, 3, 1001)[1:]
# plan_uniform = np.linspace(0, 1, 1001)[1:]

plan_linear = -np.log(np.cumprod(1-np.linspace(1e-4, 0.02, 1000)))/2
# plan_best = [0,0.1,0.3,0.5,0.7,0.9,1.1]
# run_plan(plan_best, x_range)

plt.figure(1)
plt.xlabel("steps")
plt.ylabel(r"$\overline{\alpha_t}=e^{-t}$")
plt.plot(range(1000), np.exp(-2*plan_cosine), label='cosine schedule')
plt.plot(range(1000), np.exp(-2*plan_uniform), label='uniform schedule')
plt.plot(range(1000), np.exp(-2*plan_linear), label='linear schedule')
plt.legend()
plt.savefig('weakschedule.png')
# plt.show()

plt.figure(2)
target = gaussian_kde(np.random.randn(10000)*sigma02**0.5+mu0).__call__(x_range)
plt.plot(*run_plan(plan_linear, x_range), label='linear schedule')
plt.plot(*run_plan(plan_cosine, x_range), label='cosine schedule')
plt.plot(*run_plan(plan_uniform, x_range), label='uniform schedule')
plt.plot(x_range, target, label='target distribution')
plt.legend()
# plt.show()
plt.savefig('weakresults.png')

plt.figure(3)
x = np.linspace(0, 3, 1000)
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('t')
ax1.set_ylabel('score matching loss')
ax1.plot(x, score_error(x), label='score matching loss', color=color)
ax1.legend(loc='upper left')
ax1.set_ylim(-1,26)
# ax1.tick_params(axis='y', color=color)

# Instantiate a second y-axis sharing the same x-axis
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('p.d.f of stepping schedule')
ax2.plot(x, gaussian_kde(plan_linear).__call__(x), label='linear schedule p.d.f')
ax2.plot(x, gaussian_kde(plan_cosine).__call__(x), label='cosine schedule p.d.f')
ax2.plot(x, gaussian_kde(plan_uniform).__call__(x), label='uniform schedule p.d.f')
# ax2.plot(x, gaussian_kde(plan_best).__call__(x), label='small schedule p.d.f')
# ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(-0.05, 0.8)
ax2.legend()
fig.tight_layout()  # To ensure a tight layout, preventing the right y-label from being clipped
plt.savefig('weakexplain.png')
# plt.show()