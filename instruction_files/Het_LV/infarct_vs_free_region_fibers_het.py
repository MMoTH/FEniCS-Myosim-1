# @Author: charlesmann
# @Date:   2021-12-03T12:22:13-05:00
# @Last modified by:   charlesmann
# @Last modified time: 2022-06-14T10:53:53-04:00

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Summary: Partition the ellipsoid LV into m segments longitudinally and n
# segments circumferentially, then plot the average transverse and helical angles
# from epi to endo for each segment at end diastole(?)

# Inputs:
# -------
# f0_vs_time.npy: array of shape (number_of_int_points,3,number_of_time_steps)
# that represents the f0 vector orientation for all integration points at each
# time step

# quadrature_dof.npy: of shape (number_of_int_points,3), map from integration point i
# to coordinates vector <x,y,z>

# ecc: array of shape (number_of_int_points,3) representing circumferential direction
# err: array of shape (number_of_int_points,3) representing radial direction
# ell: array of shape (number_of_int_points,3) representing longitudinal direction

# ellipsoid_deg2_norm_dist_endo.npy: normalized distance from endo (0 = on endo, 1 = on epi)

### Make a script to show how to extract the norm_dist_endo from the fiber.vtu file!

# Outputs:
# -------
# plot handles, dictionary of angles

#--------------------------------------------------------------------------------
# Load inputs:
base_dir = 'C:/Users/mme250.AD/OneDrive - University of Kentucky/Cluster_models/'
sim_dir = 'Het_model/Final_300kr/no_het_kr300/'

f0_vs_time = np.load(base_dir + sim_dir + 'f0_vs_time.npy')
quadrature_dof_map = np.load(base_dir + 'quadrature_dof.npy')
ecc = np.load(base_dir + 'ecc.npy')
err = np.load(base_dir + 'err.npy')
ell = np.load(base_dir + 'ell.npy')
norm_dist_endo = np.load(base_dir+'ellipsoid_deg2_norm_dist_endo.npy')

# Assign partition refinement
#m = 3
#n = 6

# First, set up bounds for LV partition
# Do this by looking at min and max coordinates of quadrature_dof.npy
"""highest_point = np.amax(quadrature_dof_map[:,2]) # looking at z-coordinates
lowest_point = np.amin(quadrature_dof_map[:,2])

longitudinal_partition_boundaries = np.nan*np.ones(m)
circumferential_partition_boundaries = np.nan*np.ones(n) # angles

for i in np.arange(m):
    longitudinal_partition_boundaries[i] = highest_point - (i+1)*(highest_point-lowest_point)/m

# for circumferential partition, assuming ellipsoid projected onto the x-y plane
# is centered at (0,0). First partition line will be along x >= 0
for j in np.arange(n):
    circumferential_partition_boundaries[j] = (j+1)*(360.0/n)

# Segments are numbered as follows beginning with 0:
# Start from y = 0, x >=0 at base. Traverse counter-clockwise, increasing
# the segment number whenever a circumferential partition boundary is reached.
# When the full circle has been traversed, drop down to the next longitudinal
# layer, increment the segment number. Repeat until all longitudinal partitions
# have been accounted for.

seg_list = np.linspace(0,m*n-1,m*n)"""
region_indices = np.zeros(np.shape(quadrature_dof_map)[0])
# 0 means infarct, 1 means border zone, 2 means free region

# Looping through list of integration points
# Creating an array of shape (number_of_int_points) where the value for index i
# represents which segment integration i belongs to
#int_point_seg_array = np.nan*np.ones(np.shape(quadrature_dof_map)[0])
checksum =0
checksum1 = 0
checksum2 = 0
''''
for k in np.arange(np.shape(quadrature_dof_map)[0]):
    # Getting rid of basal points?
    if quadrature_dof_map[k][2] < -.07:
        r = np.sqrt(quadrature_dof_map[k][1]**2 + (quadrature_dof_map[k][2]+.44089)**2)
        if quadrature_dof_map[k][0] > 0 and (r < .2044):
            # infarct point
            region_indices[k] = int(2)
            checksum+=1
        if quadrature_dof_map[k][0] > 0 and (r >= .2044):
            if r < (0.25):
                # border zone
                region_indices[k] = int(1)
                checksum1 +=1
            if r >= 0.25:
                if quadrature_dof_map[k][2] > (-.44089+.2044):
                    #region_indices[k] = int(5)
                    region_indices[k] = int(0)
                elif quadrature_dof_map[k][2] <= (-.44089-.2044):
                    #region_indices[k] = int(5)
                    region_indices[k] = int(0)
                else:
                    # free region at mid ventricle
                    region_indices[k] = int(0)
                    checksum2+=1
        if quadrature_dof_map[k][0] <= 0:
            if quadrature_dof_map[k][2] > (-.44089+.2044):
                #region_indices[k] = int(5)
                region_indices[k] = int(0)
            elif quadrature_dof_map[k][2] < (-.44089-.2044):
                #region_indices[k] = int(5)
                region_indices[k] = int(0)
            else:
                region_indices[k] = int(0)
                checksum2+=1
    else:
        region_indices[k] = int(5)

'''

"""print "number of infarct points", checksum
print "number of border points", checksum1
print "number of free points", checksum2
print "total number of points", np.shape(f0_vs_time)[0]"""
# Now let's try to calculate the helical angles
# For now, starting with first time point, should be -60 to 60 from epi to endo
helical_angles_init = np.nan*np.ones(np.shape(quadrature_dof_map)[0])
helical_angles_beat5 = np.nan*np.ones(np.shape(quadrature_dof_map)[0])
helical_angles_final = np.nan*np.ones(np.shape(quadrature_dof_map)[0])
transv_angles_init = np.nan*np.ones(np.shape(quadrature_dof_map)[0])
transv_angles_beat5 = np.nan*np.ones(np.shape(quadrature_dof_map)[0])
transv_angles_final = np.nan*np.ones(np.shape(quadrature_dof_map)[0])

num_cycles = 6
#mean_helical_angle_per_segment_per_cycle = np.nan*np.ones((m*n,num_cycles))
#mean_transverse_angle_per_segment_per_cycle = np.nan*np.ones((m*n,num_cycles))

for ii in np.arange(np.shape(helical_angles_init)[0]):
    temp_vec = f0_vs_time[ii,:,0]
    temp_vec2 = f0_vs_time[ii,:,-1]
    temp_vec3 = f0_vs_time[ii,:,-1]
    temp_ell = ell[ii,:]
    temp_ecc = ecc[ii,:]
    temp_err = err[ii,:]
    helical_angles_init[ii] = (180/np.pi)*np.arctan(np.dot(temp_vec,temp_ell)/np.dot(temp_vec,temp_ecc))
    helical_angles_beat5[ii] = (180/np.pi)*np.arctan(np.dot(temp_vec3,temp_ell)/np.dot(temp_vec3,temp_ecc))
    helical_angles_final[ii] = (180/np.pi)*np.arctan(np.dot(temp_vec2,temp_ell)/np.dot(temp_vec2,temp_ecc))
    transv_angles_init[ii] = (180/np.pi)*np.arctan(np.dot(temp_vec,temp_err)/np.dot(temp_vec,temp_ecc))
    transv_angles_beat5[ii] = (180/np.pi)*np.arctan(np.dot(temp_vec3,temp_err)/np.dot(temp_vec3,temp_ecc))
    transv_angles_final[ii] = (180/np.pi)*np.arctan(np.dot(temp_vec2,temp_err)/np.dot(temp_vec2,temp_ecc))

helical_angles_vs_time = np.nan*np.ones((np.shape(f0_vs_time)[0],np.shape(f0_vs_time)[2]))
transv_angles_vs_time = np.nan*np.ones((np.shape(f0_vs_time)[0],np.shape(f0_vs_time)[2]))
for i in np.arange(np.shape(f0_vs_time)[2]):
    temp = np.einsum('ij,ij->i',f0_vs_time[:,:,i],ell)
    temp2 = np.einsum('ij,ij->i',f0_vs_time[:,:,i],ecc)
    temp3 = np.einsum('ij,ij->i',f0_vs_time[:,:,i],err)
    helical_angles_vs_time[:,i] = (180/np.pi)*np.arctan(temp/temp2)
    transv_angles_vs_time[:,i] = (180/np.pi)*np.arctan(temp3/temp2)

ha_avg = np.nan*np.ones((np.shape(f0_vs_time)[0],6))
tv_avg = np.nan*np.ones((np.shape(f0_vs_time)[0],6))
ha_std = np.nan*np.ones((np.shape(f0_vs_time)[0],6))
tv_std = np.nan*np.ones((np.shape(f0_vs_time)[0],6))

# calculate averages and sd for each point each beat
for i in np.arange(6):
    for j in np.arange(np.shape(f0_vs_time)[0]):
        ha_avg[j,i] = np.average(helical_angles_vs_time[j,i*2000:(i+1)*1700])
        tv_avg[j,i] = np.average(transv_angles_vs_time[j,i*2000:(i+1)*1700])
        ha_std[j,i] = np.std(helical_angles_vs_time[j,i*2000:(i+1)*1700])
        tv_std[j,i] = np.std(transv_angles_vs_time[j,i*2000:(i+1)*1700])

ha_std_b1 = ha_std[:,0]
ha_std_b2 = ha_std[:,1]
ha_std_b3 = ha_std[:,2]
ha_std_b4 = ha_std[:,3]
ha_std_b5 = ha_std[:,4]
ha_std_b6 = ha_std[:,5]
#print "std"
#print ha_std_b1
#print ha_std_b6

tv_std_b1 = tv_std[:,0]
tv_std_b2 = tv_std[:,1]
tv_std_b3 = tv_std[:,2]
tv_std_b4 = tv_std[:,3]
tv_std_b5 = tv_std[:,4]
tv_std_b6 = tv_std[:,5]

#plt.plot(norm_dist_endo,helical_angles,'o')
#plt.ylabel('Helical Angle')
#plt.xlabel('Normalized Distance\nFrom Endo')

bin_edges = np.linspace(0,1.0,21)
right_bin_marker = np.nan*np.ones(np.shape(quadrature_dof_map)[0])

for i in np.arange(np.shape(quadrature_dof_map)[0]):
    right_bin_marker[i] = np.amin(np.where(norm_dist_endo[i] <= bin_edges))

# Let's create a data frame to handle this
d = {'right_bin_marker':right_bin_marker,'region_indices':region_indices,'helical_angles_init':helical_angles_init, \
'norm_dist_endo':norm_dist_endo,'helical_angles_final':helical_angles_final,'transv_angles_init':transv_angles_init, \
'transv_angles_final':transv_angles_final,'helical_angles_beat5':helical_angles_beat5, \
'transv_angles_beat5':transv_angles_beat5,'ha_std_b1':ha_std_b1,'ha_std_b2':ha_std_b2,'ha_std_b3':ha_std_b3,'ha_std_b4':ha_std_b4, \
'ha_std_b5':ha_std_b5,'ha_std_b6':ha_std_b6,'tv_std_b1':tv_std_b1,'tv_std_b2':tv_std_b2,'tv_std_b3':tv_std_b3, \
'tv_std_b4':tv_std_b4,'tv_std_b5':tv_std_b5,'tv_std_b6':tv_std_b6}
df = pd.DataFrame(d)

#assert(helical_angles_vs_time[:,-1].all()==d['helical_angles_final'].all())


"""fig, axs = plt.subplots(m,n,sharex='col')
fig2 = plt.figure(figsize=(4,4))
ax2 = fig2.add_subplot(111, projection='3d')
fig3,axs3 = plt.subplots(m,n,sharex='col')
fig4,axs4 = plt.subplots(m,n,sharex='col')
fig5,axs5 = plt.subplots(m,n,sharex='col')"""
fig, axs = plt.subplots(1,3)
fig2 = plt.figure(figsize=(4,4))
axs2 = fig2.add_subplot(111, projection='3d')

colors = ['r','b','k','c','g','m','w','y','#ebb734']
#colors = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18']

for p in np.arange(np.shape(quadrature_dof_map)[0]):

    q = quadrature_dof_map[p]
    axs2.scatter(q[0],q[1],q[2],zdir='z',c=colors[d['region_indices'][p].astype('int')])


"""for jj in np.arange(m):
    for kk in np.arange(n):
        axs[jj][kk].plot(d['norm_dist_endo'][d['seg_number']==jj*m+kk],d['helical_angles_final'][d['seg_number']==jj*m+kk],'o',alpha=0.25)
        #axs[jj][kk].plot(d['norm_dist_endo'][d['seg_number']==jj*m+kk],d['hel_ang_vs_time'][d['seg_number']==jj*m+kk],'o',alpha=0.25)
        #axs[jj][kk].plot(d['norm_dist_endo'][d['seg_number']==jj*m+kk],d['helical_angles_beat5'][d['seg_number']==jj*m+kk],'o',alpha=0.25)
        #axs[jj][kk].plot(d['norm_dist_endo'][d['seg_number']==jj*m+kk],d['helical_angles_init'][d['seg_number']==jj*m+kk],'o')
        axs[jj][kk].set_xlim(['endo','epi'])
        axs3[jj][kk].plot(d['norm_dist_endo'][d['seg_number']==jj*m+kk],d['transv_angles_final'][d['seg_number']==jj*m+kk],'o',alpha=0.25)
        #axs3[jj][kk].plot(d['norm_dist_endo'][d['seg_number']==jj*m+kk],d['transv_angles_beat5'][d['seg_number']==jj*m+kk],'o',alpha=0.25)
        #axs3[jj][kk].plot(d['norm_dist_endo'][d['seg_number']==jj*m+kk],d['transv_angles_init'][d['seg_number']==jj*m+kk],'o')
        axs3[jj][kk].set_xlim(['endo','epi'])
        axs4[jj][kk].plot(['beat 1','beat 2','beat 3','beat 4','beat 5','beat 6'],[np.average(d['ha_std_b1'][d['seg_number']==jj*m+kk]),np.average(d['ha_std_b2'][d['seg_number']==jj*m+kk]), \
        np.average(d['ha_std_b3'][d['seg_number']==jj*m+kk]),np.average(d['ha_std_b4'][d['seg_number']==jj*m+kk]),np.average(d['ha_std_b5'][d['seg_number']==jj*m+kk]),np.average(d['ha_std_b6'][d['seg_number']==jj*m+kk])],'o')
        axs5[jj][kk].plot(['beat 1','beat 2','beat 3','beat 4','beat 5','beat 6'],[np.average(d['tv_std_b1'][d['seg_number']==jj*m+kk]),np.average(d['tv_std_b2'][d['seg_number']==jj*m+kk]), \
        np.average(d['tv_std_b3'][d['seg_number']==jj*m+kk]),np.average(d['tv_std_b4'][d['seg_number']==jj*m+kk]),np.average(d['tv_std_b5'][d['seg_number']==jj*m+kk]),np.average(d['tv_std_b6'][d['seg_number']==jj*m+kk])],'o')"""

for jj in np.arange(3):
    #print "jj",jj
    ##print "number of points"
    #print np.shape(d['norm_dist_endo'][d['region_indices']==jj])
    #axs[jj].plot(d['norm_dist_endo'][d['region_indices']==jj],d['helical_angles_init'][d['region_indices']==jj],'o',alpha=0.25)
    #axs[jj].plot(d['norm_dist_endo'][d['region_indices']==jj],d['helical_angles_final'][d['region_indices']==jj],'o',alpha=0.25)
    for i in np.arange(np.shape(bin_edges)[0]-1):
        tempy1 = d['helical_angles_init'][(d['region_indices']==jj) & (d['right_bin_marker']==i+1)]
        tempy2 = d['helical_angles_final'][(d['region_indices']==jj) & (d['right_bin_marker']==i+1)]
        #ynew = tempy[d['right_bin_marker']==i+1]
        axs[jj].errorbar(x=(bin_edges[i+1]+bin_edges[i])/2,y=np.mean(tempy1),yerr = np.std(tempy1),marker='o',mfc='blue',mec='blue')
        axs[jj].errorbar(x=(bin_edges[i+1]+bin_edges[i])/2,y=np.mean(tempy2),yerr = np.std(tempy2),marker='o',mfc='red',mec='red')

axs[0].set_title('Free Region')
axs[1].set_title('Border Zone')
axs[2].set_title('Infarct')
axs[2].legend(['Initial Angle','Final Angle'])

axs[0].set_ylim([-70,70])
axs[1].set_ylim([-70,70])
axs[2].set_ylim([-70,70])
axs[1].set_xlabel('Normalized Distance\nFrom Endo')

axs[0].set_ylabel('Helical\nAngle',rotation=0)
#fig.suptitle('Helical Angle')
#fig3.suptitle('Transverse Angle')
#fig4.suptitle('HA STD')
#fig5.suptitle('TV STD')
#for i in np.arange(m):
    #for j in np.arange(n):
    #    axs4[i][j].set_ylim([.1,2.5])
#    axs5[i][j].set_ylim([.1,2.5])





plt.show()
