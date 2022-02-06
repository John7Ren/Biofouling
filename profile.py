from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4_3D, AdvectionRK4, ErrorCode, ParticleFile, Variable, Field, NestedField, VectorField, timer
from datetime import timedelta as delta
from datetime import  datetime
import numpy as np
import math
from glob import glob
import os
import xarray as xr
import sys
import time as timelib
import matplotlib.pyplot as plt
import warnings
import pickle
import matplotlib.ticker as mtick
import pandas as pd 
import operator
from numpy import *
import scipy.linalg
import math as math
warnings.filterwarnings("ignore")

'''Loading the Kooi theoretical profiles for physical seawater properties: 
    not time-dependent. Generated in separate python file'''

with open('profiles.pickle', 'rb') as f:
    depth,T_z,S_z,rho_z,upsilon_z,mu_z = pickle.load(f)
depth = np.array(depth)

'''Loading the Kooi theoretical profiles for biological seawater properties: 
    time-dependent. Generated in separate python file'''

with open('profiles_t.pickle', 'rb') as p:
    depth,time,A_A_t,mu_A_t = pickle.load(p)
    
# time = np.linspace(time0,total_secs,int(dt_secs+1))

#%% physical profile
# zrange = slice(0,74)
fig,ax = plt.subplots(3,3,figsize=(48,24))

ax[0,0].plot(T_z,depth,linewidth=2,c='k')
ax[0,0].invert_yaxis()
ax[0,0].set_ylim([5800,0])
ax[0,0].set_ylabel('depth [m]',fontsize=32)
ax[0,0].tick_params('both',labelsize=28)
ax[0,0].set_title('Temperature [$\degree C$]',fontsize=36)
# ax[0,0].text(0.,0.,'(a)', horizontalalignment='right',verticalalignment='bottom',fontsize=32)

ax[0,1].plot(S_z,depth,linewidth=2,c='k')
ax[0,1].invert_yaxis()
ax[0,1].set_ylim([5800,0])
# ax[0,0].set_ylabel('depth [m]',fontsize=24)
ax[0,1].tick_params('both',labelsize=28)
ax[0,1].set_title('Salinity [$kg\cdot kg^{-1}$]',fontsize=36)

ax[0,2].plot(rho_z,depth,linewidth=2,c='k')
ax[0,2].invert_yaxis()
ax[0,2].set_ylim([5800,0])
ax[0,2].set_xticks(np.arange(1023,1028.1,1),labels=np.arange(1023,1028.1,1))
ax[0,2].tick_params('both',labelsize=28)
ax[0,2].set_title('Density [$kg\cdot m^{-3}$]',fontsize=36)

ax[1,0].plot(upsilon_z,depth,linewidth=2,c='k')
ax[1,0].invert_yaxis()
ax[1,0].set_ylim([5800,0])
ax[1,0].set_ylabel('depth [m]',fontsize=32)
ax[1,0].set_xlim([0.8e-6,2.0e-6])
ax[1,0].set_xticks(np.arange(0.8e-6,2.1e-6,0.2e-6),labels=np.arange(0.8e-6,2.1e-6,0.2e-6))
ax[1,0].tick_params('both',labelsize=28)
ax[1,0].set_title('Kinematic viscosity [$m^2\cdot s^{-1}$]',fontsize=36)

ax[1,1].plot(mu_z,depth,linewidth=2,c='k')
ax[1,1].invert_yaxis()
ax[1,1].set_ylim([5800,0])
# ax[0,0].set_ylabel('depth [m]',fontsize=24)
ax[1,1].tick_params('both',labelsize=28)
ax[1,1].set_title('Dynamic viscosity [$m^2\cdot s^{-1}$]',fontsize=36)

ax[1,2].plot(mu_A_t[6,:],depth[:],linewidth=2,c='k')
ax[1,2].invert_yaxis()
ax[1,2].set_ylim([100,0])
# ax[0,0].set_ylabel('depth [m]',fontsize=24)
ax[1,2].tick_params('both',labelsize=28)
ax[1,2].set_title('Growth rate [$day^{-1}$]',fontsize=36)
ax[1,2].ticklabel_format(style='plain')

ax[2,0].plot(A_A_t[6,:],depth[:],linewidth=2,c='k')
ax[2,0].invert_yaxis()
ax[2,0].set_ylim([100,0])
ax[2,0].set_ylabel('depth [m]',fontsize=32)
ax[2,0].tick_params('both',labelsize=28)
ax[2,0].set_title('Algae concentration [$no. \cdot m^{-3}$]',fontsize=36)
ax[2,0].ticklabel_format(style='plain')

fig.delaxes(ax[2,1])
fig.delaxes(ax[2,2])

plt.text(0.9,0.9,'(a)',ha='center',va='center',fontsize=50,transform=ax[0,0].transAxes)
plt.text(0.9,0.9,'(b)',ha='center',va='center',fontsize=50,transform=ax[0,1].transAxes)
plt.text(0.9,0.9,'(c)',ha='center',va='center',fontsize=50,transform=ax[0,2].transAxes)
plt.text(0.9,0.9,'(d)',ha='center',va='center',fontsize=50,transform=ax[1,0].transAxes)
plt.text(0.9,0.9,'(e)',ha='center',va='center',fontsize=50,transform=ax[1,1].transAxes)
plt.text(0.9,0.9,'(f)',ha='center',va='center',fontsize=50,transform=ax[1,2].transAxes)
plt.text(0.9,0.9,'(g)',ha='center',va='center',fontsize=50,transform=ax[2,0].transAxes)

plt.savefig('fig_0_profile.pdf')

#%%
# zrange = slice(8,35)
zrange = slice(0,20)
aa = A_A_t[6,:]
mu_a = mu_A_t[11,:] # t from 1-11 there is a pattern of constant (1.8) above 28m (EZD), below that level mu_A=0
plt.scatter(aa[zrange],depth[zrange])
# plt.xlim([1023,1028])
# plt.plot(aa[zrange],depth[zrange])
#%% density profile - self constructed
d = 4
h = 1000
z = np.arange(0,5000,2)
rho = 1028 * ( 1 - d/(z+h) )
plt.plot(rho,-z)
plt.xlim([1023,1028])
#%% Solution to momentum equation
k = 1e-5
rho = 1000
lam1 = -(9/4)*1e-1 - 2e-5
lam2 = -(9/4)*1e-1 + 2e-5
c1 = 1 / (lam2-lam1) * rho/k*(1-1023/rho)
c2 = -(lam2-lam1+1) / (lam2-lam1) * rho/k*(1-1023/rho)
t = np.arange(0,1000,10)
z = c1*np.exp(lam1*t) + c2*np.exp(lam2*t) + rho/k*(1-1023/rho)
plt.plot(t,-z)
#%% vs and z
t = np.arange(0,100,0.1)
vs = np.zeros_like(t)
z = np.zeros_like(t)
v0 = 1/5
k_list = [1/5,1/10,1/20,1/30]

fig,ax = plt.subplots(figsize=(12,8))
for j in range(len(k_list)):
    k = k_list[j]
    t0 = np.log(v0)/(-k)
    
    for i in range(len(t)):
        if t[i]<=t0:
            vs[i] = np.exp(-k*t[i]) -v0
            z[i] = -1/k * np.exp(-k*t[i]) - v0*t[i] + 1/k
        elif t[i]>t0:
            vs[i] = -( np.exp(-k*t[i]) -v0 )
            z[i] = 1/k * np.exp(-k*t[i]) + v0*t[i] + 1/k -2/k * np.exp(-k*t[i]) - 2*v0*t[i]
    ax.plot(t,-z,label=f'{k}')
    
ax.legend()
ax.grid()