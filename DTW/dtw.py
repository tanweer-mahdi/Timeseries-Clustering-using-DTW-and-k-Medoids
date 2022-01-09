# collection of several implementations on DTW

# Naive DTW

def DTWNaive(ts1 = None, ts2 = None):
    
    DTW = {}
    s1 = np.size(ts1)
    s2 = np.size(ts2)
    
    # initializing the DTW matrix
    for i in range(s1):
        DTW[(i,-1)] = float('Inf')
    
    for j in range(s2):
        DTW[(-1,j)] = float('Inf')
        
    DTW[(-1,-1)] = 0
    
    for i in range(s1):
        for j in range(s2):
            # distance
            dist = (ts1[i] - ts2[j])**2
            # Cumulative alignment cost
            DTW[(i,j)] = dist + min(DTW[(i-1,j)],DTW[(i,j-1)],DTW[(i-1,j-1)])

    return np.sqrt(DTW[s1-1, s2-1]/(s1+s2))

# DTW with Sakoe-Chiba band

def DTWSC(ts1 = None, ts2 = None, window = None):
    
    DTW = {}
    s1 = np.size(ts1)
    s2 = np.size(ts2)
    
    w = max(np.abs(s1-s2), w)
    
    
     # initializing the DTW matrix
    for i in range(-1, s1):
        for j in range(-1, s2):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
        
    DTW[(-1,-1)] = 0
    
    for i in range(s1):
        lb = max(0, i-w)
        ub = min(s2, i+w)
        for j in range(lb,ub):
            # distance
            dist = (ts1[i] - ts2[j])**2
            # Cumulative alignment cost
            DTW[(i,j)] = dist + min(DTW[(i-1,j)],DTW[(i,j-1)],DTW[(i-1,j-1)])

    return np.sqrt(DTW[s1-1, s2-1]/(s1+s2))


# Unnormalized Keogh Lower Bound based DTW
def LB_Keogh(ts1 = None, ts2 = None, r = None):
    
    LB_sum=0
    
    s1 = np.size(ts1)
    s2 = np.size(ts2)
    
    r = max(np.abs(s1-s2), r)
    
    for ind,i in enumerate(ts1):
        
        lower_bound=min(ts2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(ts2[(ind-r if ind-r>=0 else 0):(ind+r)])
        
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    
    return np.sqrt(LB_sum)