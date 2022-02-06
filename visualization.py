import numpy as np
import math
import os
import xarray as xr
import sys
import matplotlib.pyplot as plt
import warnings
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

a_diss_rate = ''
dead = True
if dead:
    death = '_death_'
    a_diss_rate = 2.5e-8
else:
    death = ''

###############################################################################
### Visualization
###############################################################################

# load the data
path = 'outputs/'
fname = 'Kooi1D_'+str(round(simdays,2))+'d_rho'+rho_pl+'_rpl'+r_pl+death+str(a_diss_rate)+'_rhofr'+str(int(rho_fr))
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
AR['r_tot']= np.array(ds.r_tot[:,:])
AR['w']= np.array(ds.w[:,:])
AR['rho_sw']= np.array(ds.rho_sw[:,:])
AR['kin_visc']= np.array(ds.kin_visc[:,:])
AR['sw_visc']= np.array(ds.sw_visc[:,:])

therange = range(2400,2880)
# therange = range(1,len(AR['z'][0,:]))
zplt = -AR['z'][0,therange]
t = AR['time'][0,therange].astype('timedelta64[h]')
dt = 20           # days
tplt = np.arange(t[0].astype(int),t[-1].astype(int)+24*dt,24*dt).astype('float32')
tticks = np.arange(t[0].astype('timedelta64[D]').astype(int),t[-1].astype('timedelta64[D]').astype(int)+dt,dt).astype(int)

a_dead = AR['a_dead'][0,therange]
a_diss = AR['a_diss'][0,therange]
a_mort = AR['a_mort'][0,therange]
a_resp = AR['a_resp'][0,therange]
a_coll = AR['a_coll'][0,therange]
a_growth = AR['a_growth'][0,therange]
rho_tot = AR['rho_tot'][0,therange]
rho_sw = AR['rho_sw'][0,therange]
r_tot = AR['r_tot'][0,therange]
w = AR['w'][0,therange]
vs = AR['vs'][0,therange]
a = AR['a'][0,therange]
sw_visc = AR['sw_visc'][0,therange]
kin_visc = AR['kin_visc'][0,therange]

#%%
fig,ax = plt.subplots(figsize=(18,6))
ax.set_xticks(tplt, labels=tticks)
# ax.plot(t,zplt,label='depth')
# ax.plot(t,w)
# ax.plot(t,rho_sw)
# ax.plot(t,rho_tot)
# ax.plot(t,kin_visc)
ax.plot(t,a_dead,c='k',label='attached dead')
# ax.plot(t,a_diss,c='gray',label='dissolution, rate=1.0')
# ax.plot(t,tbfplt)
# ax.plot(t,dead)
# ax.plot(t,a)
# ax.plot(t,np.cumsum(-(diss-(resp+mort))))
# ax.plot(t,(resp+mort)-diss)
# ax.plot(t,rho)
# ax.grid()
# ax.set_title(fname)
# ax.set_ylim([-140,2])
# ax.set_ylabel('depth [m]',fontsize=16)
# ax.legend(loc='lower left')

# ax2 = ax.twinx()
# ax2.plot(t,vs)
# ax2.plot(t,rho_tot,c='orange')
# ax2.plot(t,diss,color='orange')
# ax2.plot(t,a_resp,c='purple',label='respiration')
# ax2.plot(t,a_mort,c='blue',label='mortality')
# ax2.plot(t,a_coll,c='orange',label='collision')
# ax2.plot(t,a_growth,c='green',label='growth')
# ax2.plot(t,a_dead,c='k',label='attached dead')
# ax2.plot(t,a_diss,c='gray',label='dissolution, rate='+str(a_diss_rate))
# ax2.plot(t,resp+mort,c='k')
# ax2.plot(t,tbfplt,color='orange')
# ax2.set_ylim([-0.02e-5,1.5e-5])
# ax2.set_ylim([-0.001,0.005])
# ax2.set_ylabel('terms [no. m^{-2}]',fontsize=16)
# ax2.legend(loc='upper left')
# plt.savefig(path+fname+'.pdf')
# plt.show()

#%% Estimate dstar
# rho_tot = 1050
# rho_sw = 1023
# kin_visc = 1e-6
# dn = 2e-4
# g = 10
# dstar = ((rho_tot - rho_sw) * g * dn**3.)/(rho_sw * kin_visc**2.)
# # w = (dstar**2.) *1.71E-4
# w = 10.**(-3.76715 + (1.92944*math.log10(dstar)) - (0.09815*math.log10(dstar)**2.) - (0.00575*math.log10(dstar)**3.) + (0.00056*math.log10(dstar)**4.))