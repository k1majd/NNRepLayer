import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.interpolate import interp1d

def loadData(name_csv):
    with open(name_csv) as csv_file:
        data = np.asarray(list(csv.reader(csv_file, delimiter=',')), dtype=np.float32)
    return data

# GET PHASE FROM DATA
# n=20364

# with open('demos/walking_example/data/heel_strikes.csv') as csv_file:
#     data = np.asarray(list(csv.reader(csv_file, delimiter=',')), dtype=np.float32)

# phase=np.array([]).reshape(0,)
# for i in range(data.shape[0]-1):
#     temp = np.linspace(0,1,int(data[i+1]-data[i])+1)
#     phase=np.concatenate((phase,temp[:-1]))
#     pass
# np.savetxt("demos/walking_example/data/phase.csv", phase, delimiter=",")


# SET LIMITS
Dfem = loadData('demos/walking_example/data/GeoffFTF_1.csv')
Dtib = loadData('demos/walking_example/data/GeoffFTF_2.csv')
Dfut = loadData('demos/walking_example/data/GeoffFTF_3.csv')
phase = loadData('demos/walking_example/data/GeoffFTF_phase.csv')
n=20364
Dankle = np.subtract(Dtib[:n,1], Dfut[:n,1])
observations = np.concatenate((Dfem[:n,1:], Dtib[:n,1:]), axis=1)
observations = (observations - observations.mean(0))/observations.std(0)
controls = Dankle  #(Dankle - Dankle.mean(0))/Dankle.std(0)
phase = phase[:n]

lim_x=np.array([ 0.00,   0.075,   0.12,  0.30,  0.40,  0.46,  0.485, 0.56,   0.625,  0.675,  0.78,   0.885,  0.98,   1])
lim_u=np.array([ 9,     3.0,     4.0,   13,    18,    22.0,  23.0,  15.0,   0.0,    1.0,    9.0,    4.5,    11,      9])+1
lim_l=np.array([ 0,      -5,      -4.,   6.0,   11,    14.0,  15,    -2.0,   -14,    -11,    -1.0,   -6.5,   1,      0])-.8

new_x = np.linspace(0,1,51)
lim_uc = interp1d(lim_x, lim_u, kind='cubic')
lim_lc = interp1d(lim_x, lim_l, kind='cubic')

lim_u = np.interp(new_x,lim_x,lim_u)
lim_l = np.interp(new_x,lim_x,lim_l)

plt.scatter(phase[:],controls[:], alpha=.1)
plt.plot(new_x, lim_uc(new_x), '--r')
plt.plot(new_x, lim_lc(new_x), '--r')
# plt.plot(new_x, lim_u, 'k')
# plt.plot(new_x, lim_l, 'k')
plt.grid()
plt.show()

limits = np.concatenate((new_x.reshape(-1,1), lim_uc(new_x).reshape(-1,1), lim_lc(new_x).reshape(-1,1)),axis=1)

# np.savetxt("demos/walking_example/data/GeoffFTF_limits.csv", limits, delimiter=",")





