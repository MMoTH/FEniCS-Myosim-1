import matplotlib
import matplotlib.pyplot as plt
import sys
import numpy as np
#plt.style.use('seaborn')


base_dir = 'C:/Users/mme250.AD/OneDrive - University of Kentucky/Cluster_models/'
sim_dir = 'Het_model/Final_time_step0.1/not_het4/'

Num_model = 4
CS = 2000

fig, (ax1)  = plt.subplots(nrows=1,ncols=1)
ax1.set_xlabel('Volume (mL)',fontsize=13)
ax1.set_ylabel('LV Pressure (mmHg)',fontsize=13)

for jj in np.arange(1,Num_model+1):

    if jj == 1:
        pv_file_input= 'C:/Users/mme250.AD/OneDrive - University of Kentucky/Cluster_models/Het_model/Final_time_step0.1/not_het4/PV_.txt'
    if jj == 2:
        pv_file_input= 'C:/Users/mme250.AD/OneDrive - University of Kentucky/Cluster_models/Het_model/Final_time_step0.1/het4_0.1fib/PV_.txt'
    if jj == 3:
        pv_file_input= 'C:/Users/mme250.AD/OneDrive - University of Kentucky/Cluster_models/Het_model/Final_time_step0.1/het4_0.2fib/PV_.txt'
    if jj == 4:
        pv_file_input= 'C:/Users/mme250.AD/OneDrive - University of Kentucky/Cluster_models/Het_model/Final_time_step0.1/het4_0.3fib/PV_.txt'
    

    #pv_file_input = sys.argv[1]
    #pv_file_input = np.load(base_dir + sim_dir + 'PV_.txt')
    #pv_file_input = 'PV_.txt'

    time, LV_pressure, arterial_pressure, venous_pressure, LV_vol, ven_vol, art_vol = np.loadtxt(pv_file_input, skiprows=0, unpack=True, usecols=(0,1,2,3,4,5,6))
    #time, LV_pressure, LV_vol, ven_vol, art_vol = np.loadtxt(pv_file_input, skiprows=0, unpack=True, usecols=(0,1,2,3,4))
    calcium = np.zeros(np.shape(time))



    #ax1.plot(LV_vol,LV_pressure,)
    #ax1.plot(LV_vol[31:200],LV_pressure[31:200],'*')
    ax1.plot(LV_vol[-CS:-1],LV_pressure[-CS:-1],)
    plt.xlim([0.05, .275])
    plt.ylim([0, 125])
ax1.legend(['No het','10 percent het','20 percent het','30 percent het'],loc='upper right')
#ax1.legend(loc='upper right')

fig.tight_layout(pad=0.5)


#plt.show()




#fig, (ax1)  = plt.subplots(nrows=1,ncols=1)


fig, ax1 = plt.subplots(2,1,sharex='col',squeeze=False)
ax1[0][0].set_xlabel('time_step')
ax1[0][0].set_ylabel('LV Pressure (mmHg)',fontsize=18)
ax1[1][0].set_ylabel('Volume (mL)',fontsize=18)
ax1[0][0].grid(color='b', linestyle='--', linewidth=.3,axis='x')
ax1[1][0].grid(color='b', linestyle='--', linewidth=.3,axis='x')

for jj in np.arange(1,Num_model+1):
    

    if jj == 1:
        pv_file_input= 'C:/Users/mme250.AD/OneDrive - University of Kentucky/Cluster_models/Het_model/Final_time_step0.1/not_het4/PV_.txt'
    if jj == 2:
        pv_file_input= 'C:/Users/mme250.AD/OneDrive - University of Kentucky/Cluster_models/Het_model/Final_time_step0.1/het4_0.1fib/PV_.txt'
    if jj == 3:
        pv_file_input= 'C:/Users/mme250.AD/OneDrive - University of Kentucky/Cluster_models/Het_model/Final_time_step0.1/het4_0.2fib/PV_.txt'
    if jj == 4:
        pv_file_input= 'C:/Users/mme250.AD/OneDrive - University of Kentucky/Cluster_models/Het_model/Final_time_step0.1/het4_0.3fib/PV_.txt'
    

    #pv_file_input = sys.argv[1]
    #pv_file_input = np.load(base_dir + sim_dir + 'PV_.txt')
    #pv_file_input = 'PV_.txt'

    time, LV_pressure, arterial_pressure, venous_pressure, LV_vol, ven_vol, art_vol = np.loadtxt(pv_file_input, skiprows=0, unpack=True, usecols=(0,1,2,3,4,5,6))
    
    ax1[0][0].plot(LV_pressure,)
    ax1[1][0].plot(LV_vol,)
    #ax1.plot(LV_vol[31:200],LV_pressure[31:200],'*')
    #ax1.plot(LV_vol[-CS:-1],LV_pressure[-CS:-1],)
    #plt.xlim([0.05, .275])
    #plt.ylim([0, 125])
ax1[0][0].legend(['No het','10 percent het','20 percent het','30 percent het'],loc='upper right')
plt.show()






'''
#time = data[:,0]
#pressure = data[:,1]
#volume = data[:2]


fig, (ax1, ax2, ax3)  = plt.subplots(nrows=3,ncols=1)
ax1.plot(LV_vol,LV_pressure)
ax1.set_xlabel('Volume (mL)')
ax1.set_ylabel('LV Pressure (mmHg)')

ax2.plot(time,LV_pressure,label='LV Pressure')
ax2.plot(time,arterial_pressure,label='Arterial Pressure')
ax2.plot(time,venous_pressure,label='Venous Pressure')

#ax2.legend()
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.175), shadow=True, ncol=3)

#ax2.set_xlabel('Time (ms)')
ax2.legend(loc='lower left')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Pressure (mmHg)')
ax4 = ax2.twinx()
color= 'tab:gray'
ax4.set_ylabel('calcium',color=color)
ax4.plot(time,calcium,color=color)
ax4.tick_params(axis='y',labelcolor=color)

l1 = ax3.plot(time,LV_vol,label='LV Volume')
l3 = ax3.plot(time,art_vol,label='Arterial Volume')
l2 = ax3.plot(time,ven_vol,label='Venous Volume')

ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.175), shadow=True, ncol=3)

#l2 = ax3.plot(time,art_vol,label='Arterial Volume')
#l3 = ax3.plot(time,ven_vol,label='Venous Volume')
#ax3.legend((l1, l2, l3),('LV Volume','Venous Volume','Arterial Volume'))
#ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Volume (mL)')
#ax3.legend()
ax3.legend(loc='lower left')

fig.tight_layout(pad=0.5)
'''


