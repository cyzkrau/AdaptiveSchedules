import numpy as np
import random
import matplotlib.pyplot as plt
import json

def GA(plan, smb, L, C, eta):
    g = [0, ]
    for i in range(1, len(plan)-1):
        t1, t2, t3 = plan[i-1], plan[i], plan[i+1]
        l = smb(t3)*(t3-t2)+smb(t2)*(t2-t1)+C*L(t3)*(t3-t2)**2+C*L(t2)*(t2-t1)**2
        t2 += 1e-5
        lpdl = smb(t3)*(t3-t2)+smb(t2)*(t2-t1)+C*L(t3)*(t3-t2)**2+C*L(t2)*(t2-t1)**2
        g.append((lpdl-l)*1e5)
    g.append(0)
    # print(g)
    for i in range(len(plan)):
        plan[i] -= g[i]*eta
    return plan

def EB(plan, smb, L, C):
    se = 0
    for i in range(1, len(plan)):
        tk1, tk2 = plan[i-1], plan[i]
        se += smb(tk2) * (tk2-tk1) + C*L(tk2) * (tk2-tk1)**2
    return se

def run_KL(plan, score_error, sigma02, mu0):
    # KL(P_0,Q_T)
    sigmat2 = lambda t: np.exp(-2*t) * (sigma02 - 1) + 1
    mut = lambda t: np.exp(-t) * mu0
    score = lambda x,t: -(x-mut(t))/sigmat2(t)

    m, s = 0, 1
    for k in range(len(plan)-1,0,-1):
        dt = plan[k] - plan[k-1]
        m += (m+2*score(m,plan[k])+2*score_error(plan[k]))*dt
        s = (1 + (1 - 2/sigmat2(plan[k]))*dt )**2 + 2*dt

    # Yts = np.random.randn(1000000)
    # for k in range(len(plan)-1,0,-1):
    #     dt = plan[k] - plan[k-1]
    #     Yts += (Yts + 2*score(Yts, plan[k])) * dt + np.sqrt(2*dt) * np.random.randn(len(Yts))
    #     Yts += 2*score_error(plan[k]) * dt
    # m, s = Yts.mean(), Yts.var()
        
    return np.log(s / sigma02) + ((sigma02**2 + (mu0 - m)**2) / (2 * s**2)) - 0.5
    
if __name__ == '__main__':

    plan = np.linspace(0,3,10)

    # lightly score error - epoch200,eta1e-3
    num, eta, epoch, score_error = 2, 1e-3, 100, lambda t: 1./(1.1-np.exp(-2*t)) + np.cos(2*t*np.pi) + 1
    # heavily shaking score error - epoch20,eta1e-5
    # num, eta, epoch, score_error = 1, 1e-5, 20, lambda t: 10*np.cos(10*t*np.pi) + 10

    plt.figure(1)
    plt.xlabel('time')
    plt.ylabel('score matching error')
    # fig, ax = plt.subplots()
    xs = np.linspace(0,3,100000)
    plt.plot(xs, score_error(xs), label='score matching error', c='y')
    plt.plot(plan, [score_error(i) for i in plan], label='initial discretization points', c='r')
    for i in plan:
        plt.scatter(i, score_error(i), c='r')
    ori_plan = plan
        
    sigma02=0.16
    mu0=10
    C = 1.
    L=lambda t:(1-np.exp(-2*t)*(1-sigma02))**(-2)
    
    losss = [EB(plan, smb=lambda t:score_error(t)**2, L=L, C=C)]
    kls = [run_KL(plan, score_error, sigma02, mu0)]
    for _ in range(epoch):
        plan = GA(list(plan), smb=lambda t:score_error(t)**2, L=L, C=C, eta=eta)
        losss.append(EB(plan, smb=lambda t:score_error(t)**2, L=L, C=C))
        kls.append(run_KL(plan, score_error, sigma02, mu0))
    plt.plot(plan, [score_error(i) for i in plan], label=f'discretization points after {epoch} iterations GA', c='g')
    # plt.ylim(-2, 27)

    for i in plan:
        plt.scatter(i, score_error(i), c='g')
    plt.legend(loc='upper right')
    
    plt.savefig(f'showGAshakepoint{num}.png')
    # plt.show()


    plt.figure(2)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('$KL(P_0\|Q_T)$')
    ax1.plot(range(1+epoch), kls, c='b', label='$KL(P_0\|Q_T)$')
    for i in range(1+epoch):
        ax1.scatter(i, kls[i], c='b')
    ax1.legend(loc='upper right',bbox_to_anchor=(1, 1))
    # ax1.set_ylim(-1,26)
    # ax1.tick_params(axis='y', color=color)

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('EB')
    ax2.plot(range(1+epoch), losss, c='r', label='EB')
    for i in range(1+epoch):
        ax2.scatter(i, losss[i], c='r')
    # ax2.set_ylim(-0.05, 0.8)
    ax2.legend(loc='upper right',bbox_to_anchor=(1, 0.92))
    fig.tight_layout()  # To ensure a tight layout, preventing the right y-label from being clipped
    plt.savefig(f'showGAshakeKL{num}.png')

