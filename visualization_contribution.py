import numpy as np
import math
import os
import xarray as xr
import sys
import matplotlib.pyplot as plt
import warnings
from datetime import timedelta
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
warnings.filterwarnings("ignore")


###############################################################################
### Parameters
###############################################################################

#------ CHOOSE (Note: the same values must also be placed in the Kooi kernel: lines 53 and 54) -----
rho_pl = "920"                 # density of plastic (kg m-3): DEFAULT FOR FIG 1: 920 but full range is: 840, 920, 940, 1050, 1380 (last 2 are initially non-buoyant)
r_pl = "1e-04"                # radius of plastic (m): DEFAULT FOR FIG 1: 10-3 to 10-6 included but full range is: 10 mm to 0.1 um or 10-2 to 10-7

lon = np.array([-161,-159]) #lon release locations
lat = np.array([35,37]) #lat release locations
simdays =  150 #number of days running the sim
secsdt = 60 #timestep of sim

time0 = 0
secsoutdt = 60*60 # seconds in an hour (must be in hours due to algal pickle profiles being hours)
total_secs = secsoutdt*24.*simdays - secsoutdt # total time (in seconds) being run for the sim
dt_secs = total_secs/secsoutdt # time steps

rho_bf = 1388.              # density of biofilm ([g m-3]
rho_fr = 2600.              # density of frustule [g m-3], Amaral-Zettler, L.A. et al. 2021
v_a = 2.0E-16 
r_a = ((3./4.)*(v_a/math.pi))**(1./3.)      # radius of algae [m]
r_cy = 59./60. * r_a                        # radius of cytosplasm [m]
v_cy = (4./3.)*math.pi*r_cy**3.             # volume of cytoplasm [m3]
v_fr = v_a - v_cy                           # volume of frustule [m3]

EZD = -28

a_diss_rate = ''
dead = True
if dead:
    death = '_death_'
    a_diss_rate = 2.5e-4
else:
    death = ''

###############################################################################
### Visualization
###############################################################################

# load the data
path = 'outputs/'
fname = 'Kooi1D_'+str(round(simdays,2))+'d_rho'+rho_pl+'_rpl'+r_pl+'_rhofr'+str(int(rho_fr))
ds = xr.open_dataset(path+fname+'.nc')
print(fname)

AR = {}
AR['z'] = np.array(ds.z[:,:])
AR['aa'] = np.array(ds.aa[:,:]) * (v_a*rho_bf)
AR['time'] = np.array(ds.time[:,:])
AR['vs'] = np.array(ds.vs[:,:])
AR['a'] = np.array(ds.a[:,:]) * (v_a*rho_bf)
AR['a_dead'] = np.array(ds.a_dead[:,:]) * (v_fr*rho_fr)
AR['a_diss'] = np.array(ds.a_diss[:,:])
AR['a_mort'] = np.array(ds.a_mort[:,:])
AR['a_resp'] = np.array(ds.a_resp[:,:])
AR['a_coll'] = np.array(ds.a_coll[:,:])
AR['a_growth'] = np.array(ds.a_growth[:,:])
AR['rho_tot']= np.array(ds.rho_tot[:,:])
AR['rho_sw']= np.array(ds.rho_sw[:,:])

# therange = range(1200,1440)
therange = range(1,len(AR['z'][0,:]))
t = AR['time'][0,therange].astype('timedelta64[h]')
dt = 10           # days

tplt = np.arange(t[0].astype(int),t[-1].astype(int)+24*dt,24*dt).astype('float32')
tticks = np.arange(t[0].astype('timedelta64[D]').astype(int),t[-1].astype('timedelta64[D]').astype(int)+dt,dt).astype(int)
zplt = -AR['z'][0,therange]
# t = AR['time'][0,therange]
a_dead = AR['a_dead'][0,therange]
a_diss = AR['a_diss'][0,therange]
a_mort = AR['a_mort'][0,therange]
a_resp = AR['a_resp'][0,therange]
a_coll = AR['a_coll'][0,therange]
a_growth = AR['a_growth'][0,therange]
rho = AR['rho_tot'][0,therange]
rhosw = AR['rho_sw'][0,therange]
a = AR['a'][0,therange]

#%%
fname = 'Kooi1D_'+str(round(simdays,2))+'d_rho'+rho_pl+'_rpl'+r_pl+'_death_'+str(a_diss_rate)+'_rhofr'+str(int(rho_fr))
ds = xr.open_dataset(path+fname+'.nc')
print(fname)

AR = {}
AR['z'] = np.array(ds.z[:,:])
AR['aa'] = np.array(ds.aa[:,:]) * (v_a*rho_bf)
AR['time'] = np.array(ds.time[:,:])
AR['vs'] = np.array(ds.vs[:,:])
AR['a'] = np.array(ds.a[:,:]) * (v_a*rho_bf)
AR['a_dead'] = np.array(ds.a_dead[:,:]) * (v_fr*rho_fr)
AR['a_diss'] = np.array(ds.a_diss[:,:])
AR['a_mort'] = np.array(ds.a_mort[:,:])
AR['a_resp'] = np.array(ds.a_resp[:,:])
AR['a_coll'] = np.array(ds.a_coll[:,:])
AR['a_growth'] = np.array(ds.a_growth[:,:])
AR['rho_tot']= np.array(ds.rho_tot[:,:])
AR['rho_sw']= np.array(ds.rho_sw[:,:])

# therange = range(700,1000)
# therange = range(0,len(AR['z'][0,:]))
zplt_2d = -AR['z'][0,therange]
a_dead_2d = AR['a_dead'][0,therange]
a_diss_2d = AR['a_diss'][0,therange]
a_mort_2d = AR['a_mort'][0,therange]
a_resp_2d = AR['a_resp'][0,therange]
a_coll_2d = AR['a_coll'][0,therange]
a_growth_2d = AR['a_growth'][0,therange]
rho_2d = AR['rho_tot'][0,therange]
rhosw_2d = AR['rho_sw'][0,therange]
a_2d = AR['a'][0,therange]


tot_growth = np.array([np.sum(a_growth_2d),np.sum(a_growth)])
tot_resp = np.array([np.sum(a_resp_2d),np.sum(a_resp)])
tot_coll = np.array([np.sum(a_coll_2d),np.sum(a_coll)])
tot_mort = np.array([np.sum(a_mort_2d),np.sum(a_mort)])
tot_diss = np.array([np.sum(a_diss_2d),np.sum(a_diss)])

norm = tot_growth + tot_resp + tot_coll + tot_mort + tot_diss
norm_growth = tot_growth/norm * 100
norm_resp = tot_resp/norm * 100
norm_coll = tot_coll/norm * 100
norm_mort = tot_mort/norm * 100
norm_diss = tot_diss/norm * 100

tot_dead = np.array([np.sum(a_dead_2d),np.sum(a_dead)])
tot_living = np.array([np.sum(a_2d),np.sum(a)])
norm_acc = tot_dead+tot_living
norm_dead = tot_dead / norm_acc * 100
norm_living = tot_living / norm_acc * 100
# norm_dead = np.zeros(2)
# norm_living = np.zeros(2)
# norm_dead[0] = norm_resp[0] + norm_mort[0] + norm_diss[0]
# norm_living = norm_growth + norm_coll + norm_resp + norm_mort - norm_dead
# tot_rho = [np.sum(rho_2d),np.sum(rho)]

#%% Contribution - bar
xlabels = ['With dead', 'No dead']
x = np.array([1,1.4])
# Bar parameters
width = 0.15
width0 = 0.2
width1 = 0.4
# Plot
fig,ax = plt.subplots(figsize=(16,12))
# Positive terms
b1 = ax.bar(x+width*0.5,norm_dead,width,label='Dead',color='k',alpha=1.)
b2 = ax.bar(x-width*0.5,norm_living,width,label='Living',color='tab:cyan',alpha=1.)
# b3 = ax.bar(x-width*1.5,norm_coll,width,label='Collision',color='tab:orange')
# b4 = ax.bar(x-width*1.0,norm_growth,width,label='Growth',color='tab:green',)
# b5 = ax.bar(x+width*0.5,norm_resp,width,label='Respiration',color='tab:olive')
# b6 = ax.bar(x+width*0.5,norm_mort,width,label='Mortality',color='tab:brown',bottom=norm_resp)
# ax.bar(x+width*0.5,norm_resp,bottom=norm_mort)
# Negative terms
# ax.bar(x-width*0.5,-norm_resp,width,label='Respiration',color='tab:olive')
# ax.bar(x-width*0.5,-norm_mort,width,label='Mortality',color='tab:brown',bottom=-norm_resp)
# b7 = ax.bar(x+width*1.5,-norm_diss,width,label='Dissolution',color='tab:gray')

ax.tick_params('both',labelsize=18)
ax.set_ylabel('Portion [%]',fontsize=20)
ax.set_xlim([0.8,1.6])
ax.set_xlabel('Situation',fontsize=20)
ax.legend(fontsize=20,loc='best')
# ax.set_ylabel('Instant terms [$kg \cdot m^{-2}$]',fontsize=20,)
ax.set_xlabel('Situations',fontsize=20)
ax.tick_params(axis='y',labelsize=18)
ax.set_xticks(x,xlabels,fontsize=18)
ax.set_ylim([0,110])
ax.set_title('Portion of dead and living cells',fontsize=24,fontweight='bold')
# ax.legend(handles=[b1,b2,b3,b4,b5,b6,b7],fontsize=20,loc='best')
ax.grid()

# ax0.bar(x,tot_rho,width1,label='Dead',color='red',alpha=0.05)
# ax0.tick_params('both',labelsize=18)
# ax0.set_xlim([0.75,1.85])
# ax0.set_xlabel('Situation',fontsize=20)
# plt.style.use('ggplot')
# plt.grid('on')
plt.savefig('fig_2_fraction_withTitle.pdf')

#%% Contribution - bar - other option
xlabels = ['With dead', 'No dead']
x = np.array([1,1.6])
# Bar parameters
width = 0.1
width0 = 0.2
width1 = 0.4
# Plot
fig,ax0 = plt.subplots(figsize=(16,12))
ax = ax0.twinx()
ax1 = ax0.twinx()
ax1.spines['left'].set_visible(True)
ax1.yaxis.set_label_position('left')
ax1.yaxis.set_ticks_position('left')
ax0.set_yticks([])

b1 = ax1.bar(x+width0*0.5,tot_dead,width0,label='Dead',color='k',alpha=1.,zorder=2.6)
b2 = ax1.bar(x-width0*0.5,tot_living,width0,label='Living',color='tab:cyan',alpha=1.)
ax1.tick_params('both',labelsize=18)
ax1.set_ylabel('Accumulative terms [$kg \cdot m^{-2}$]',fontsize=20)
ax1.set_xlim([0.75,1.85])
ax1.set_xlabel('Situation',fontsize=20)
ax1.legend(fontsize=20,loc='upper left')

b3 = ax.bar(x-width*1.5,tot_coll,width,label='Collision',color='tab:orange')
b4 = ax.bar(x-width*0.5,tot_growth,width,label='Growth',color='tab:green',)
b5 = ax.bar(x+width*0.5,tot_resp,width,label='Respiration',color='tab:olive')
b6 = ax.bar(x+width*1.5,tot_mort,width,label='Mortality',color='tab:brown')
b7 = ax.bar(x+width*2.5,tot_diss,width,label='Dissolution',color='tab:gray')
ax.legend(fontsize=20,loc='upper right')
# ax.set_yscale('log')

ax.set_ylabel('Instant terms [$kg \cdot m^{-2}$]',fontsize=20,)
ax.set_xlabel('Situations',fontsize=20)
ax.tick_params(axis='y',labelsize=18)
ax.set_xticks(x,xlabels,fontsize=18)
ax.set_ylim([0,0.016])
ax.set_title('Contribution of different terms',fontsize=24,fontweight='bold')
# ax.legend(handles=[b1,b2,b3,b4,b5,b6,b7],fontsize=20)
# ax.grid()

ax0.bar(x,tot_rho,width1,label='Dead',color='red',alpha=0.05)
ax0.tick_params('both',labelsize=18)
ax0.set_xlim([0.75,1.85])
ax0.set_xlabel('Situation',fontsize=20)

# plt.savefig('contribution1.pdf')

#%% Contribution - line
fig,ax = plt.subplots(2,1,figsize=(16,20))

# No Dead
ax2 = ax[0].twinx()
ax3 = ax[0].twinx()
ax[0].plot(t,rho,label='Total density - No dead',c='tab:blue')
ax[0].plot(t,rhosw,label='Sea water density',c='k',alpha=0.5)
ax[0].grid(which='major')
ax[0].set_ylim([920,1070])
ax[0].set_yticks(np.arange(1000,1061,20))
ax[0].set_ylabel('Total density [$kg\cdot m^{-3}$]',fontsize=20)
ax[0].legend(loc='upper left',fontsize=16)
ax[0].set_title('No dead situation',fontsize=24)
ax[0].set_xlabel('Time [d]',fontsize=20)
ax[0].set_xlim([t[0].astype(int),t[-1].astype(int)+1])
ax[0].set_xticks(tplt, labels=tticks,fontsize=16)
ax[0].tick_params('both',labelsize=16)
# ax[0].axhline(y=EZD,c='gray')

ax2.plot(t,a,c='tab:cyan',label='Attached living')
ax2.plot(t,a_dead,c='k',label='Attached dead')
ax2.set_ylim([-0.0001,0.0300])
ax2.set_yticks(np.arange(0,0.016,0.003))
ax2.set_ylabel('Accumulative terms mass [kg]', fontsize=20)
ax2.legend(loc='lower left',fontsize=16)
ax2.tick_params('both',labelsize=16)
ax2.grid()

# ax3.plot(t,a_coll,c='tab:orange',label='collision')
# ax3.plot(t,a_growth,c='tab:green',label='growth')
# ax3.plot(t,a_resp,c='tab:olive',label='respiration')
# ax3.plot(t,a_mort,c='tab:brown',label='mortality')
# ax3.plot(t,a_diss,c='tab:gray',label='dissolution')


# Dead
ax2 = ax[1].twinx()
ax3 = ax[1].twinx()
ax[1].plot(t,rho_2d,label='Total density - With dead',c='tab:red')
ax[1].plot(t,rhosw_2d,label='Sea water density',c='k',alpha=0.5)
ax[1].grid(which='major')
ax[1].set_ylim([920,1070])
ax[1].set_yticks(np.arange(1000,1061,20))
ax[1].set_ylabel('Total density [$kg\cdot m^{-3}$]',fontsize=20)
ax[1].legend(loc='upper left',fontsize=16)
ax[1].set_title('With dead situation',fontsize=24)
ax[1].set_xlabel('Time [d]',fontsize=20)
ax[1].set_xlim([t[0].astype(int),t[-1].astype(int)+1])
ax[1].set_xticks(tplt, labels=tticks,fontsize=16)
ax[1].tick_params('both',labelsize=16)
# ax[1].axhline(y=EZD,c='gray')

ax2.plot(t,a_2d,c='tab:cyan',label='Attached living')
ax2.plot(t,a_dead_2d,c='k',label='Attached dead')
ax2.set_ylim([-0.0001,0.0300])
ax2.set_yticks(np.arange(0,0.016,0.003))
ax2.set_ylabel('Accumulative terms mass [kg]', fontsize=20)
ax2.legend(loc='lower left',fontsize=16)
ax2.tick_params('both',labelsize=16)
ax2.grid()

# ax3.plot(t,a_coll_2d,c='tab:orange',label='collision')
# ax3.plot(t,a_growth_2d,c='tab:green',label='growth')
# ax3.plot(t,a_resp_2d,c='tab:olive',label='respiration')
# ax3.plot(t,a_mort_2d,c='tab:brown',label='mortality')
# ax3.plot(t,a_diss_2d,c='tab:gray',label='dissolution')

plt.tight_layout()
# plt.suptitle('Total density and accumulative terms',fontsize=24,fontweight='bold')
# plt.savefig('contribution_line.pdf')

