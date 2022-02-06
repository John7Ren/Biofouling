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
simdays =  720 #number of days running the sim
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
    a_diss_rate = 0.0
else:
    death = ''

###############################################################################
### Visualization
###############################################################################
path = 'outputs/'
fname = 'Kooi1D_'+str(round(simdays,2))+'d_rho'+rho_pl+'_rpl'+r_pl+'_death_'+str(a_diss_rate)
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
# therange = range(1200,1441)
therange = range(1,len(AR['z'][0,:]))
t = AR['time'][0,therange].astype('timedelta64[h]')
dt = 30           # days

tplt = np.arange(t[0].astype(int),t[-1].astype(int)+24*dt,24*dt).astype('float32')
tticks = np.arange(t[0].astype('timedelta64[D]').astype(int),t[-1].astype('timedelta64[D]').astype(int)+dt,dt).astype(int)
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
fig,ax = plt.subplots(1,1,figsize=(16,8))

# ax.plot(t,zplt,label='Vertical trajectory - No dead',c='tab:blue')
ax.plot(t,zplt_2d,label='Vertical trajectory - With dead',c='tab:blue',alpha=1)
ax.grid(which='major')
ax.set_ylim([-320,10])
ax.set_ylabel('depth [m]',fontsize=14)
ax.legend(loc='upper left')
ax.set_title('1D trajectories and total density changes in 150 days',fontsize=16)
ax.set_xlabel('time [d]',fontsize=14)
# ax.set_xlim([t[0].astype(int),t[-1].astype(int)+1])
ax.set_xticks(tplt, labels=tticks)
# ax[0].axhline(y=EZD,c='gray')
# ax.set_yticks(np.arange(-60,1,10))
ax2 = ax.twinx()
# ax2.plot(t,rho,label='Total density - No dead',c='tab:blue')
ax2.plot(t,rho_2d,label='Total density - With dead',c='tab:red',alpha=1)
ax2.plot(t,rhosw_2d,c='k')
ax2.set_ylim([900,1200])
ax2.legend(loc='lower left')
# ax2.set_yticks(np.arange(920,1061,20))
ax2.grid()
