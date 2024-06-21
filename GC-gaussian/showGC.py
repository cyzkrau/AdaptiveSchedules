import numpy as np
import random
import matplotlib.pyplot as plt

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

def GC(points, plan, smb, L, C, max_step_length=1000, adjust_step=1000):
    # # plan: t-based, forward
    
    # adjust
    for m in range(1,len(plan)-1):
        tk1, tk3 = plan[m-1], plan[m+1]
        best, best_tk2 = -1, plan[m]
        for tk2 in range(plan[m]-adjust_step, plan[m]+adjust_step):
            if tk2<=tk1 or tk2 >= tk3:
                continue
            if tk2-tk1>max_step_length or tk3-tk2>max_step_length:
                continue
            if m==1 and tk2>max_step_length/2:
                continue
            se = smb[tk3] * (points[tk3]-points[tk2]) + smb[tk2] * (points[tk2]-points[tk1])
            de = L(points[tk3]) * (points[tk3]-points[tk2])**2 + L(points[tk2]) * (points[tk2]-points[tk1])**2
            if best > se+C*de or best < 0:
                best, best_tk2 = se+C*de, tk2
        plan[m] = best_tk2
    
    return plan

def EB(points, plan, smb, L, C):
    se = 0
    for i in range(1, len(plan)):
        tk1, tk2 = plan[i-1], plan[i]
        se += smb[tk2] * (points[tk2]-points[tk1]) + C*L(points[tk2]) * (points[tk2]-points[tk1])**2
    return se

if __name__ == '__main__':
    n, sep = 51, 3
    mu0, sigma02 = 10, 0.16
    points = np.linspace(0,1,n+1)

    # random error
    score_error = [1/(1+i/len(points))+random.random()*0.4+1 for i in range(len(points))]
    plan = list(range(0,n+1,sep))
    plt.figure(1)
    plt.xlabel('time')
    plt.ylabel('score matching error')
    plt.plot(points, score_error, label='all available discretization points', c='y')
    plt.plot([points[i] for i in plan], [score_error[i] for i in plan], label='normal plan', c='r')
    for i in plan:
        plt.scatter(points[i], score_error[i], c='r')
        
    EBs = [EB(points, plan, smb=np.array(score_error)**2, L=lambda t:(1-np.exp(-2*t)*(1-sigma02))**(-2), C=10.),]
    # print(plan)
    kls = [run_KL([points[i] for i in plan], lambda t:score_error[int(t*n)], sigma02, mu0),]
    for _ in range(5):
        plan = GC(points, plan, smb=np.array(score_error)**2, L=lambda t:(1-np.exp(-2*t)*(1-sigma02))**(-2), C=10., adjust_step=17)
        EBs.append(EB(points, plan, smb=np.array(score_error)**2, L=lambda t:(1-np.exp(-2*t)*(1-sigma02))**(-2), C=10.))
        kls.append(run_KL([points[i] for i in plan], lambda t:score_error[int(t*n)], sigma02, mu0))

    plt.plot([points[i] for i in plan], [score_error[i] for i in plan], label='GC 5iteration', c='g')
    for i in plan:
        plt.scatter(points[i], score_error[i], c='g')
    plt.legend()
    plt.savefig('showGC.png')
    # plt.show()

    plt.figure(2)
    fig, ax1 = plt.subplots()
    ax1.plot(range(6), EBs, c='r', label='EB')
    for i in range(6):
        ax1.scatter(i, EBs[i], c='r')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('EB')
    ax1.legend(loc='upper right',bbox_to_anchor=(1, 1))
    # plt.legend()
    # plt.savefig('showGCloss1.png')
    # plt.show()

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('$KL(P_0\|Q_T)$')
    ax2.plot(range(6), kls, c='b', label='$KL(P_0\|Q_T)$')
    for i in range(6):
        ax2.scatter(i, kls[i], c='b')
    # ax2.set_ylim(-0.05, 0.8)
    ax2.legend(loc='upper right',bbox_to_anchor=(1, 0.92))
    fig.tight_layout()  # To ensure a tight layout, preventing the right y-label from being clipped
    plt.savefig('showGCloss.png')



