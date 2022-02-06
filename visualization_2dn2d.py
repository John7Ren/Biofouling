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


#%%

###############################################################################
### Visualization
###############################################################################

# load the data
path = '/Users/renjiongqiu/Biofouling-SOAC/outputs/'
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

therange = range(2401,3601)
# therange = range(1,len(AR['z'][0,:3601]))
t = AR['time'][0,therange].astype('timedelta64[h]')
dt = 5           # days

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

#%% Fig. 1
day0 = tticks[0]
dayz = tticks[-1]
fig,ax = plt.subplots(1,1,figsize=(16,12))

ax.plot(t,zplt,label='Vertical trajectory - No dead',c='tab:blue')
ax.plot(t,zplt_2d,label='Vertical trajectory - With dead',c='tab:red',alpha=1.)
ax.grid(which='major')
ax.set_ylim([-140,2])
ax.set_ylabel('Depth [m]',fontsize=20)
ax.legend(loc='upper left',fontsize=16)
ax.set_title(f'1D trajectories and total density changes from day {day0} to {dayz}',fontsize=24,fontweight='bold')
ax.set_xlabel('Time [d]',fontsize=20)
ax.set_xlim([t[0].astype(int),t[-1].astype(int)+1])
ax.set_xticks(tplt, labels=tticks,fontsize=16)
# ax[0].axhline(y=EZD,c='gray')
ax.set_yticks(np.arange(-60,1,10),fontsize=16)
ax.tick_params('both',labelsize=14)
ax2 = ax.twinx()
ax2.plot(t,rho,label='Total density - No dead',c='tab:blue')
ax2.plot(t,rho_2d,label='Total density - With dead',c='tab:red',alpha=1.)
ax2.plot(t,rhosw,c='k',label='Sea water density')
# ax2.set_ylim([1000,1080]) # rho_pl = 1000
ax2.set_ylim([920,1200])
ax2.legend(loc='lower left',fontsize=16)
ax2.set_yticks(np.arange(920,1061,20),fontsize=16)
ax2.set_ylabel('Total density [$kg\cdot m^{-3}$]',fontsize=20)
ax2.tick_params('both',labelsize=16)
ax2.grid()

# plt.savefig('sens_diss1.pdf')

#%% Fig. 1 - Animation
t_data = []
zplt_2d_data = []
rho_2d_data = []
zplt_data = []
rho_data = []

fig,ax = plt.subplots(1,1,figsize=(16,12))

line1z, = ax.plot(t,zplt,label='Vertical trajectory - No dead',c='tab:blue')
line0z, = ax.plot(t,zplt_2d,label='Vertical trajectory - With dead',c='tab:red')
ax.grid(which='major')
ax.set_ylim([-140,2])
ax.set_ylabel('Depth [m]',fontsize=20)
ax.legend(loc='upper left',fontsize=16)
ax.set_title('1D trajectories and total density changes from day 100 to 150',fontsize=24,fontweight='bold')
ax.set_xlabel('Time [d]',fontsize=20)
# ax.set_xlim([t[0].astype(int),t[-1].astype(int)+1])
ax.set_xticks(tplt, labels=tticks,fontsize=16)
# ax[0].axhline(y=EZD,c='gray')
ax.set_yticks(np.arange(-60,1,10),fontsize=16)
ax.tick_params('both',labelsize=16)
ax2 = ax.twinx()
line0r, = ax2.plot(t,rho_2d,label='Total density - With dead',c='tab:red')
line1r, = ax2.plot(t,rho,label='Total density - No dead',c='tab:blue')
ax2.plot(t,rhosw,c='k',label='Sea water density')
ax2.set_ylim([920,1200])
ax2.legend(loc='lower left',fontsize=16)
ax2.set_yticks(np.arange(920,1061,20),fontsize=16)
ax2.set_ylabel('Total density [$kg\cdot m^{-3}$]',fontsize=20)
ax2.tick_params('both',labelsize=16)
ax2.grid()

def animate(i):
    t_data.append(t[i])
    zplt_2d_data.append(zplt_2d[i])
    rho_2d_data.append(rho_2d[i])
    zplt_data.append(zplt[i])
    rho_data.append(rho[i])
    
    line0z.set_data(t_data,zplt_2d_data)
    line0r.set_data(t_data,rho_2d_data)
    line1z.set_data(t_data,zplt_data)
    line1r.set_data(t_data,rho_data)
    # scatter0.set_offsets(np.c_[t[i].astype(int),zplt_2d[i]])
    return line0z,line0r,line1z,line1r,

anim = FuncAnimation(fig, animate, frames=len(t), interval=10)
anim.save('anim1_zVSrho.gif', writer='imagemagick', fps=15)

#%% Fig. 1 - Other option 1
day0 = tticks[0]
dayz = tticks[-1]
fig,ax = plt.subplots(1,2,figsize=(32,12))

ax[0].plot(t,zplt,label='Vertical trajectory',c='tab:blue')
# ax[0].plot(t,zplt_2d,label='Vertical trajectory - With dead',c='tab:red',alpha=1.)
ax[0].grid(which='major')
ax[0].set_ylim([-140,2])
ax[0].set_ylabel('Depth [m]',fontsize=28)
ax[0].legend(loc='upper left',fontsize=28)
ax[0].set_title('No dead situation', fontsize=36)
ax[0].set_xlabel('Time [d]',fontsize=28)
ax[0].set_xlim([t[0].astype(int),t[-1].astype(int)+1])
ax[0].set_xticks(tplt, labels=tticks,fontsize=24)
# ax[0].axhline(y=EZD,c='gray')
ax[0].set_yticks(np.arange(-60,1,10),fontsize=24)
ax[0].tick_params('both',labelsize=22)
ax2 = ax[0].twinx()
ax2.plot(t,rho,label='Total density',c='tab:blue')
# ax2.plot(t,rho_2d,label='Total density - With dead',c='tab:red',alpha=1.)
ax2.plot(t,rhosw,c='k',label='Sea water density')
# ax2.set_ylim([1000,1080]) # rho_pl = 1000
ax2.set_ylim([920,1200])
ax2.legend(loc='lower left',fontsize=28)
ax2.set_yticks(np.arange(920,1061,20),fontsize=24)
# ax2.set_ylabel('Total density [$kg\cdot m^{-3}$]',fontsize=28)
ax2.tick_params('both',labelsize=22)
ax2.grid()

# ax[1].plot(t,zplt,label='Vertical trajectory - No dead',c='tab:blue')
ax[1].plot(t,zplt_2d,label='Vertical trajectory',c='tab:red',alpha=1.)
ax[1].grid(which='major')
ax[1].set_ylim([-140,2])
# ax[1].set_ylabel('Depth [m]',fontsize=28)
ax[1].legend(loc='upper left',fontsize=28)
ax[1].set_title('With dead situation', fontsize=36)
ax[1].set_xlabel('Time [d]',fontsize=28)
ax[1].set_xlim([t[0].astype(int),t[-1].astype(int)+1])
ax[1].set_xticks(tplt, labels=tticks,fontsize=24)
# ax[0].axhline(y=EZD,c='gray')
ax[1].set_yticks(np.arange(-60,1,10),fontsize=24)
ax[1].tick_params('both',labelsize=22)
ax2 = ax[1].twinx()
# ax2.plot(t,rho,label='Total density - No dead',c='tab:blue')
ax2.plot(t,rho_2d,label='Total density',c='tab:red',alpha=1.)
ax2.plot(t,rhosw,c='k',label='Sea water density')
# ax2.set_ylim([1000,1080]) # rho_pl = 1000
ax2.set_ylim([920,1200])
ax2.legend(loc='lower left',fontsize=28)
ax2.set_yticks(np.arange(920,1061,20),fontsize=24)
ax2.set_ylabel('Total density [$kg\cdot m^{-3}$]',fontsize=28)
ax2.tick_params('both',labelsize=22)
ax2.grid()

plt.text(0.95,0.95,'(a)',ha='center',va='center',fontsize=40,transform=ax[0].transAxes)
plt.text(0.95,0.95,'(b)',ha='center',va='center',fontsize=40,transform=ax[1].transAxes)

# plt.suptitle(f'1D trajectories and total density changes from day {day0} to {dayz}',fontsize=24,fontweight='bold',position=(0.5,0.92))
# plt.tight_layout(w_pad=0.)

plt.savefig('fig_1_oscillation.pdf')
#%% Fig. 1 - Other option 2
fig,ax = plt.subplots(2,1,figsize=(18,24))
# Dead
# ax[0].plot(t,zplt,label='No death')
ax[0].plot(t,zplt_2d,label='Vertical trajectory')
ax[0].grid(which='major')
ax[0].set_ylim([-140,2])
ax[0].set_ylabel('depth [m]',fontsize=14)
ax[0].legend(loc='upper left')
ax[0].set_title('With dead situation',fontsize=16)
ax[0].set_xlabel('time [d]',fontsize=14)
ax[0].set_xlim([t[0].astype(int),t[-1].astype(int)+1])
ax[0].set_xticks(tplt, labels=tticks)
# ax[0].axhline(y=EZD,c='gray')
ax[0].set_yticks(np.arange(-60,1,10))
ax2 = ax[0].twinx()
ax2.plot(t,rho_2d,c='tab:orange',label='Total density')
ax2.plot(t,rhosw_2d,c='gray')
ax2.set_ylim([920,1200])
ax2.legend(loc='lower left')
ax2.set_yticks(np.arange(920,1061,20))
ax2.grid()

# No Dead
ax[1].plot(t,zplt,label='Vertical trajectory')
# ax[0].plot(t,zplt_2d,label='With death')
ax[1].grid(which='major')
ax[1].set_ylim([-140,2])
ax[1].set_ylabel('depth [m]',fontsize=14)
ax[1].legend(loc='upper left')
ax[1].set_title('No dead situation',fontsize=16)
ax[1].set_xlabel('time [d]',fontsize=14)
ax[1].set_xticks(tplt, labels=tticks)
ax[1].set_xlim([t[0].astype(int),t[-1].astype(int)+1])
ax[1].axhline(y=EZD,c='gray')
ax[1].set_yticks(np.arange(-60,1,10))
ax2 = ax[1].twinx()
ax2.plot(t,rho,c='tab:orange',label='Total density')
ax2.plot(t,rhosw,c='gray')
ax2.set_ylim([920,1200])
ax2.legend(loc='lower left')
ax2.set_yticks(np.arange(920,1061,20))
ax2.grid()

# plt.savefig('test.pdf')
#%% Fig. 1 - Animation (other option)
# animdt = timedelta(days=1)

fig,ax = plt.subplots(2,1,figsize=(18,24))
t_data = []
zplt_2d_data = []
rho_2d_data = []
zplt_data = []
rho_data = []
# Dead
# ax[0].plot(t,zplt,label='No death')
line0z, = ax[0].plot(t[0],zplt_2d[0],label='Vertical trajectory',c='tab:red')
# scatter0 = ax[0].scatter(t[0].astype(int),zplt_2d[0])
ax[0].grid(which='major')
ax[0].set_ylim([-140,2])
ax[0].set_xlim([t[0].astype(int),t[-1].astype(int)])
ax[0].set_ylabel('depth [m]',fontsize=14)
ax[0].legend(loc='upper left')
ax[0].set_title('With dead situation',fontsize=16)
ax[0].set_xlabel('time [d]',fontsize=14)
ax[0].set_xticks(tplt, labels=tticks)
# ax[0].axhline(y=EZD,c='gray')
ax2 = ax[0].twinx()
line0r, = ax2.plot(t[0],rho_2d[0],label='Total density',c='tab:red')
ax2.plot(t,rhosw_2d,c='k')
ax2.set_ylim([920,1200])
ax2.legend(loc='lower left')

# No Dead
line1z, = ax[1].plot(t[0],zplt[0],label='Vertical trajectory',c='tab:blue')
# ax[0].plot(t,zplt_2d,label='With death')
ax[1].grid(which='major')
ax[1].set_ylim([-140,2])
ax[1].set_xlim([t[0].astype(int),t[-1].astype(int)])
ax[1].set_ylabel('depth [m]',fontsize=14)
ax[1].legend(loc='upper left')
ax[1].set_title('No dead situation',fontsize=16)
ax[1].set_xlabel('time [d]',fontsize=14)
ax[1].set_xticks(tplt, labels=tticks)
# ax[1].axhline(y=EZD,c='gray')
ax2 = ax[1].twinx()
line1r, = ax2.plot(t[0],rho[0],label='Total density',c='tab:blue')
ax2.plot(t,rhosw,c='k')
ax2.set_ylim([920,1200])
ax2.legend(loc='lower left')

def animate(i):
    t_data.append(t[i])
    zplt_2d_data.append(zplt_2d[i])
    rho_2d_data.append(rho_2d[i])
    zplt_data.append(zplt[i])
    rho_data.append(rho[i])
    
    line0z.set_data(t_data,zplt_2d_data)
    line0r.set_data(t_data,rho_2d_data)
    line1z.set_data(t_data,zplt_data)
    line1r.set_data(t_data,rho_data)
    # scatter0.set_offsets(np.c_[t[i].astype(int),zplt_2d[i]])
    return line0z,line0r,line1z,line1r,

anim = FuncAnimation(fig, animate, frames=len(t), interval=10)
anim.save('anim1_zVSrho_2.gif', writer='imagemagick', fps=15)
# anim = FuncAnimation(fig, animate, interval=10)
# writer = FFMpegWriter(fps=15)
# anim.save('anim.mp4',writer=writer)
'''
fig,ax = plt.subplots(3,1,figsize=(18,24))
# Depth
ax[0].plot(t,zplt,label='No death')
ax[0].plot(t,zplt_2d,label='With death')
ax[0].grid(which='major')
ax[0].set_ylim([-70,2])
ax[0].set_ylabel('depth [m]',fontsize=14)
ax[0].legend(loc='lower left')
ax[0].set_title('1D vertical trajectories',fontsize=16)
ax[0].set_xlabel('time [d]',fontsize=14)
ax[0].set_xticks(tplt, labels=tticks)

# No death
# ax[1].plot(t,rho,label='rho')
ax[1].plot(t,a_resp,c='purple',label='respiration')
ax[1].plot(t,a_mort,c='blue',label='mortality')
ax[1].plot(t,a_coll,c='orange',label='collision')
# ax[1].plot(t,a_growth,c='green',label='growth')
# ax[1].set_yscale('log')
# ax[1].plot(t,a_diss,c='gray',label='dissolution')
ax[1].set_ylabel('terms [no.]',fontsize=14)
ax[1].legend(loc='upper left')
# ax[1].set_ylim([-0.2e7,13e7])
ax[1].grid()
ax[1].set_xticks(tplt, labels=tticks)
ax[1].set_title('Contributions of different processes - No dead',fontsize=16)
ax[1].set_xlabel('time [d]',fontsize=14)
# ax2=ax[1].twinx()
# ax2.plot(t,a,c='darkred',label='attached living')
# ax2.plot(t,a_dead,c='k',label='attached dead')
# ax2.legend()
# ax2.set_ylim([-0.0001,0.0250])


# With death
# ax[2].plot(t,rho_2d,label='rho')
ax[2].plot(t,a_resp_2d,c='purple',label='respiration')
ax[2].plot(t,a_mort_2d,c='blue',label='mortality')
ax[2].plot(t,a_coll_2d,c='orange',label='collision')
ax[2].plot(t,a_growth_2d,c='green',label='growth')
# ax[2].set_yscale('log')
# ax[2].plot(t,a_diss_2d,c='gray',label='dissolution, rate='+str(a_diss_rate))
ax[2].set_ylabel('terms [no.]',fontsize=14)
ax[2].legend(loc='upper left')
# ax[2].set_ylim([-0.2e7,10e7])
ax[2].grid()
ax[2].set_xticks(tplt, labels=tticks)
ax[2].set_title('Contributions of different processes - With dead',fontsize=16)
ax[2].set_xlabel('time [d]',fontsize=14)
ax2=ax[2].twinx()
ax2.plot(t,a_2d,c='darkred',label='attached living')
ax2.plot(t,a_dead_2d,c='k',label='attached dead')
ax2.legend()
ax2.set_ylim([-0.0001,0.0200])
'''

# # Rho
# ax[3].plot(t,rho,label='respiration')
# ax[3].plot(t,rho_2d,label='mortality')
# # ax[2].plot(t,a_coll_2d,c='orange',label='collision')
# # ax[2].plot(t,a_growth_2d,c='green',label='growth')
# # ax[2].plot(t,a_diss_2d,c='gray',label='dissolution, rate='+str(a_diss_rate))
# # ax[2].set_ylabel('terms [no.]',fontsize=14)
# # ax[2].legend(loc='upper left')
# ax[3].set_ylim([900,1100])
# ax[3].grid()
# ax[3].set_xticks(tplt, labels=tticks)
# # ax[2].set_title('Contributions of different processes - With dead',fontsize=16)
# # ax[2].set_xlabel('time [d]',fontsize=14)
# # ax2=ax[2].twinx()
# # ax2.plot(t,a_2d,c='darkred',label='attached living')
# # ax2.plot(t,a_dead_2d,c='k',label='attached dead')
# # ax2.legend()
# # ax2.set_ylim([-0.0001,0.0250])

# plt.savefig(path+'2dn2d.pdf')
# plt.show()