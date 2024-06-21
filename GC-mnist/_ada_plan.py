import numpy as np

def ada_choose_plan(points, plan, smb, L, C, max_num=None, max_step_length=1000, adjust_step=5):
    # plan: t-based, forward
    plan.insert(0, -1)
    smb.insert(0, -1)
    np.append(points, 0)
    max_num = len(plan) if max_num is None else max_num
    
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
    
    del smb[0]
    return plan[1:]