import numpy as np
import math
import os
import xarray as xr
import sys
import matplotlib.pyplot as plt
import warnings
from datetime import timedelta
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
dt_secs = int(total_secs/secsoutdt+2) # time steps

# rho_bf = 1388.              # density of biofilm ([g m-3]
rho_cy = 1325.4
rho_fr = 2600.              # density of frustule [g m-3], Amaral-Zettler, L.A. et al. 2021
v_a = 2.0E-16 
r_a = ((3./4.)*(v_a/math.pi))**(1./3.)      # radius of algae [m]
r_cy = 59./60. * r_a                        # radius of cytosplasm [m]
v_cy = (4./3.)*math.pi*r_cy**3.             # volume of cytoplasm [m3]
v_fr = v_a - v_cy                           # volume of frustule [m3]
# rho_bf = ( v_fr*rho_fr + v_cy*rho_cy) / v_a

a_diss_rate = ''
dead = False
if dead:
    death = '_death_'
    a_diss_rate_list = [0.0, 2.5e-8, 2.5e-7, 2.5e-6, 2.5e-5, 2.5e-4, 2.5e-3, 2.5e-2, 2.5e-1, 1.0]
    # a_diss_rate_list = [2.5e-4, 2.5e-3, 2.5e-2, 2.5e-1, 1.0]
    # a_diss_rate_list = [0.0, 2.5e-8, 2.5e-7, 2.5e-6, 2.5e-5, 2.5e-4]
else:
    death = ''

change_rho_fr = True
if change_rho_fr:
    death = '_death_'
    a_diss_rate = 0.0
    rho_fr_list = [1325., 1800., 2600.]


#%% Importing data
path = 'outputs/'
# therange = range(1080,1120)
therange = range(1,dt_secs)
variable = rho_fr_list
N = len(variable)

zplt = np.zeros((len(therange),N))
a = np.zeros((len(therange),N))
aa = np.zeros((len(therange),N))
a_dead = np.zeros((len(therange),N))
a_diss = np.zeros((len(therange),N))
a_mort = np.zeros((len(therange),N))
a_resp = np.zeros((len(therange),N))
a_growth = np.zeros((len(therange),N))
a_coll = np.zeros((len(therange),N))
rho_tot = np.zeros((len(therange),N))
rho_sw = np.zeros((len(therange),N))
rho_delta = np.zeros((len(therange),N))
vs = np.zeros((len(therange),N))

for i,rho_fr in enumerate(variable):
    # rho_bf
    rho_bf = ( v_fr*rho_fr + v_cy*rho_cy) / v_a
    # diss_rate
    # fname = 'Kooi1D_'+str(round(simdays,2))+'d_rho'+rho_pl+'_rpl'+r_pl+death+str(rate)+'_rhofr'+str(int(rho_fr))
    # rho_fr
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
    # AR['rho_sw']= np.array(ds.rho_sw[:,:])

    t = AR['time'][0,therange].astype('timedelta64[h]')
    dt = 50           # days
    tplt = np.arange(t[0].astype(int),t[-1].astype(int)+24*dt,24*dt).astype('float32')
    tticks = np.arange(t[0].astype('timedelta64[D]').astype(int),t[-1].astype('timedelta64[D]').astype(int)+dt,dt).astype(int)
    
    zplt[:,i] = -AR['z'][0,therange]
    a_dead[:,i] = AR['a_dead'][0,therange]
    a_diss[:,i] = AR['a_diss'][0,therange]
    a_mort[:,i] = AR['a_mort'][0,therange]
    a_resp[:,i] = AR['a_resp'][0,therange]
    a_coll[:,i] = AR['a_coll'][0,therange]
    a_growth[:,i] = AR['a_growth'][0,therange]
    a[:,i] = AR['a'][0,therange]
    aa[:,i] = AR['aa'][0,therange]
    rho_tot[:,i] = AR['rho_tot'][0,therange]
    # rho_sw[:,i] = AR['rho_sw'][0,therange]
    rho_delta[:,i] = rho_sw[:,i] - rho_tot[:,i]
    vs[:,i] = AR['vs'][0,therange]
    
tot_growth = np.sum(a_growth,axis=0)
tot_resp = np.sum(a_resp,axis=0)
tot_coll = np.sum(a_coll,axis=0)
tot_mort = np.sum(a_mort,axis=0)
tot_diss = np.sum(a_diss,axis=0)
tot_dead = np.sum(a_dead,axis=0)
tot_living = np.sum(a,axis=0)
tot_rho = np.sum(rho_tot,axis=0)
norm_acc = tot_dead+tot_living
norm_dead = tot_dead / norm_acc * 100
norm_living = tot_living / norm_acc * 100
mean_rho = np.mean(rho_tot,axis=0)

#%% Depth and rho
###############################################################################
### Visualization
###############################################################################
# pltrange=range(5,10)
pltrange=range(0,6)
panel = len(pltrange)
colors=['black','purple','olive','tab:orange','tab:green','tab:red','tab:brown','tab:pink','tab:cyan','tab:blue']
labels=variable
fig,ax = plt.subplots(panel,3,figsize=(24,4.5*panel))

for i,index in enumerate(pltrange):
    for j in range(3):
        if j == 0:
            ax[i,j].plot(t,zplt[:,index],label='Vertical trajectory',c=colors[index])
            ax[i,j].grid(which='major')
            ax[i,j].set_ylim([-200,2])
            ax[i,j].set_ylabel('depth [m]',fontsize=14)
            ax[i,j].legend(loc='best')
            ax[i,j].set_title(f'Trajectories - {variable[index]}',fontsize=16)
            ax[i,j].set_xlabel('time [d]',fontsize=14)
            # ax[i,j].set_xlim([t[0].astype(int),t[-1].astype(int)//7])
            ax[i,j].set_xticks(tplt, labels=tticks)
            # ax[0].axhline(y=EZD,c='gray')
            ax[i,j].set_yticks(np.arange(-200,1,50))
            
        elif j == 1:
            # No Dead
            ax[i,j].plot(t,rho_tot[:,index],label='Total density',c=colors[index])
            ax[i,j].grid(which='major')
            ax[i,j].set_ylim([980,1060])
            ax[i,j].set_ylabel('[$kg\cdot m^{-3}$]',fontsize=14)
            ax[i,j].legend(loc='best')
            ax[i,j].set_title(f'Total density - {variable[index]}',fontsize=16)
            ax[i,j].set_xlabel('time [d]',fontsize=14)
            # ax[i,j].set_xlim([t[0].astype(int),t[-1].astype(int)//7])
            ax[i,j].set_xticks(tplt, labels=tticks)
            # ax[i,j].plot(y=EZD,c='k')
            # ax[i,j].set_yticks(np.arange(980,1060,40))
        elif j == 2:
            # No Dead
            ax[i,j].plot(t,a[:,index],label='Attached living',c=colors[index])
            ax[i,j].plot(t,a_dead[:,index],label='Attached dead',c=colors[index],linestyle='--')
            ax[i,j].grid(which='major')
            ax[i,j].set_ylim([-0.0001,0.025])
            ax[i,j].set_ylabel('[$kg \cdot m^{-2}$]',fontsize=14)
            ax[i,j].legend(loc='best')
            ax[i,j].set_title(f'Accumulative terms - {variable[index]}',fontsize=16)
            ax[i,j].set_xlabel('time [d]',fontsize=14)
            # ax[i,j].set_xlim([t[0].astype(int),t[-1].astype(int)//7])
            ax[i,j].set_xticks(tplt, labels=tticks)
            # ax[i,j].plot(y=EZD,c='k')
            ax[i,j].set_yticks(np.arange(0.000,0.026,0.005))
            
            # Other terms
            # ax2 = ax[i,j].twinx()
            # ax2.plot(t,a_growth[:,index],c='tab:green',label='growth')
            # ax2.plot(t,a_resp[:,index],c='tab:olive',label='respiration')
            # ax2.plot(t,a_mort[:,index],c='tab:brown',label='mortality')
            # ax2.plot(t,a_diss[:,index],c='tab:gray',label='dissolution')
            # ax2.set_yscale('log')

plt.tight_layout()
# plt.savefig('sensitivity_analysis4.pdf')

#%% Depth and rho - other option 1
###############################################################################
### Visualization
###############################################################################
# pltrange=range(5,10)
pltrange=range(0,3)
panel = len(pltrange)
colors=['black','purple','olive','tab:orange','tab:green','tab:red','tab:brown','tab:pink','tab:cyan','tab:blue']
text = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)']
# colors=['black','purple','tab:red']
labels=variable
fig,ax = plt.subplots(panel,1,figsize=(12,2*panel))

for i,index in enumerate(pltrange):
        
        # depth
        ax[i].plot(t,zplt[:,index],label='Vertical trajectory',c=colors[index])
        ax[i].set_ylim([-200,2])
        ax[i].set_yticks(np.arange(-200,1,50))
        ax[i].set_ylabel('depth [m]',fontsize=14)
        # # accumulative terms
        # ax[i].plot(t,a[:,index],label='Attached living',c=colors[index])
        # ax[i].plot(t,a_dead[:,index],label='Attached dead',c='k',linestyle='--')
        # ax[i].set_ylim([-0.0001,0.015])
        # # ax[i].set_yticks(np.arange(-200,1,50))
        # ax[i].set_ylabel('[$kg \cdot m^{-2}$]',fontsize=14)
        
        ax[i].grid(which='major')
        # ax[i].legend(loc='upper right')
        ax[i].set_title(f'Dissolution rate: {variable[index]} $hr^{{-1}}$',fontsize=16)
        # ax[i].set_title(f'Density of frustule: {variable[index]} $kg\cdot m^{{-3}}$',fontsize=16)
        ax[i].set_xlim([t[0].astype(int),t[-1].astype(int)])
        ax[i].set_xticks(tplt, labels=tticks)
        # ax[0].axhline(y=EZD,c='gray')
        
        if i ==len(pltrange)-1:
            ax[i].set_xlabel('time [d]',fontsize=14)
        
        plt.tight_layout(pad=2)
        plt.text(0.03,0.15,text[i],ha='center',va='center',fontsize=20,transform=ax[i].transAxes)
        
# plt.suptitle('Trajectories of setting different dissolution rate',fontsize=20,fontweight='bold',position=(0.5,0.94))
# plt.tight_layout(pad=2)
# plt.savefig('fig_5_sens_rhofr.pdf')
plt.savefig('fig_5_sens_rhofr.pdf')
#%% Contribution - bar
pltrange = slice(0,-1)
# pltrange = slice(5,10)
xlabels = a_diss_rate[pltrange]
x = np.arange(len(xlabels))

tot_growth_plt = tot_growth[pltrange]
tot_resp_plt = tot_resp[pltrange]
tot_coll_plt = tot_coll[pltrange]
tot_mort_plt = tot_mort[pltrange]
tot_diss_plt = tot_diss[pltrange]
tot_dead_plt = tot_dead[pltrange]
tot_living_plt = tot_living[pltrange]
tot_rho_plt = tot_rho[pltrange]
norm_dead_plt = norm_dead[pltrange]
norm_living_plt = norm_living[pltrange]

# Bar parameters
width = 0.1
width0 = 0.2
width1 = 0.4
# Plot
fig,ax0 = plt.subplots(figsize=(16,12))
ax = ax0.twinx()
ax1 = ax0.twinx()
ax.set_zorder(3)
ax1.spines['left'].set_visible(True)
ax1.yaxis.set_label_position('left')
ax1.yaxis.set_ticks_position('left')
ax0.set_yticks([])

b1 = ax1.bar(x+width0*0.5,tot_dead_plt,width0,label='Dead',color='k',alpha=1.,zorder=2.6)
b2 = ax1.bar(x-width0*0.5,tot_living_plt,width0,label='Living',color='tab:cyan',alpha=1.)
ax1.tick_params('both',labelsize=18)
ax1.set_ylabel('Accumulative terms [$kg \cdot m^{-2}$]',fontsize=20)
# ax1.set_xlim([0.75,1.85])
ax1.set_xlabel('Situation',fontsize=20)
# ax1.legend(fontsize=20,loc='upper left')

b3 = ax.bar(x-width*1.5,tot_coll_plt,width,label='Collision',color='tab:orange')
b4 = ax.bar(x-width*0.5,tot_growth_plt,width,label='Growth',color='tab:green')
b5 = ax.bar(x+width*0.5,tot_resp_plt,width,label='Respiration',color='tab:olive',alpha=0.5)
b6 = ax.bar(x+width*1.5,tot_mort_plt,width,label='Mortality',color='tab:brown',alpha=0.5)
b7 = ax.bar(x+width*2.5,tot_diss_plt,width,label='Dissolution',color='tab:gray')
# ax.legend(fontsize=20,loc='upper right')
# ax.set_yscale('log')

ax.set_ylabel('Instant terms [$kg \cdot m^{-2}$]',fontsize=20,)
ax.set_xlabel('Situations',fontsize=20)
ax.tick_params(axis='y',labelsize=18)
ax.set_xticks(x,xlabels,fontsize=18)
# ax.set_ylim([0,0.016])
ax.set_title('Contribution of different terms',fontsize=24,fontweight='bold')
ax.legend(handles=[b1,b2,b3,b4,b5,b6,b7],fontsize=20)
# ax.grid()

ax0.bar(x,tot_rho_plt,width1,label='Dead',color='red',alpha=0.05)
ax0.tick_params('both',labelsize=18)
# ax0.set_xlim([0.75,1.85])
ax0.set_xlabel('Situation',fontsize=20)

# plt.savefig('sensitivity_contribution2.pdf')

#%% Contribution - bar - other option 1
pltrange = slice(0,10)
# pltrange = slice(5,10)
xlabels = a_diss_rate_list[pltrange]
x = np.arange(len(xlabels))

norm_dead_plt = norm_dead[pltrange]
norm_living_plt = norm_living[pltrange]

# Bar parameters
width = 0.3
# Plot
fig,ax = plt.subplots(figsize=(26,12))
# Positive terms
ax.bar(x+width*0.5,norm_dead_plt,width,label='Dead',color='k',alpha=1.)
ax.bar(x-width*0.5,norm_living_plt,width,label='Living',color='tab:cyan',alpha=1.)

ax.tick_params('both',labelsize=24)
ax.set_ylabel('Portion [%]',fontsize=28)
ax.set_xlim([-0.5,9.5])
ax.set_xlabel('Dissolution rate [$hr^{-1}$]',fontsize=28)
ax.legend(fontsize=28,loc='upper left')
# ax.tick_params(axis='y',labelsize=22)
ax.set_xticks(x,xlabels,fontsize=24)
ax.set_ylim([0,110])
# ax.set_title('Portion of dead and living cells',fontsize=24,fontweight='bold')
# ax.set_title('Portion of death and living cells',fontsize=24,fontweight='bold')
ax.grid()

plt.savefig('fig_3_sens_portion.pdf')

#%% Averaged total density
pltrange = slice(0,6)
# pltrange = slice(5,10)
xlabels = a_diss_rate_list[pltrange]
x = np.arange(len(xlabels))

mean_rho_plt = mean_rho[pltrange]

# Plot
fig,ax = plt.subplots(figsize=(16,12))
# Positive terms
ax.plot(x,mean_rho_plt,marker='s',markersize=12,c='k')
ax.axhline(y=1023,c='k',linestyle='--',linewidth=3.5)

ax.tick_params('both',labelsize=18)
ax.set_ylabel('Total density [$kg\cdot m^{-3}$]',fontsize=20)
# ax.set_xlim([-0.5,9.5])
ax.set_xlabel('Dissolution rate [$hr^{-1}$]',fontsize=20)
# ax.legend(fontsize=20,loc='best')
ax.tick_params(axis='y',labelsize=18)
ax.set_xticks(x,xlabels,fontsize=18)
# ax.set_ylim([0,110])
ax.set_title('Averaged total density',fontsize=24,fontweight='bold')
# ax.set_title('Portion of death and living cells',fontsize=24,fontweight='bold')
ax.grid()

plt.savefig('fig_6_sens_rho_part1.pdf')

#%% Other option
pltrange=range(0,6)
colors=['black','purple','blue','orange','green','red','chocolate']
labels=a_diss_rate
fig,ax = plt.subplots(4,1,figsize=(18,16))
for i in pltrange:
    ax[0].plot(t,zplt[:,i],c=colors[i],label=labels[i])
    ax[0].legend()
    ax[0].set_title('depth')
    # ax[0].axhline(y=-28,c=colors[i])
    ax2 = ax[0].twinx()
    # ax2.plot(t,rho_tot[:,i],c='blue',label=labels[i])
    # ax2.plot(t,rho_sw[:,i],c='blue')
    ax2.legend()
    ax2.set_ylim([1000,1050])
    
    # # Dissolution
    # ax[1].plot(t,a_diss[:,i],c=colors[i],label=labels[i])
    # ax[1].legend()
    # ax[1].set_title('dissolution')
    
    # Dissolution
    ax[1].plot(t,rho_tot[:,i],c=colors[i],label=labels[i])
    # ax[1].plot(t,rho_sw[:,i])
    ax[1].legend()
    # ax[1].set_ylim([1000,1050])
    ax[1].set_title('total density')
    ax2 = ax[1].twinx()
    # ax2.plot(t,vs[:,i],c='k')
    # ax2.set_ylim([-0.001,0.005])
    
    ax[2].plot(t,a_dead[:,i],c=colors[i],label=labels[i])
    ax[2].set_yscale('log')
    ax[2].legend()
    ax[2].set_title('attached dead')
    ax2 = ax[2].twinx()
    # ax2.plot(t,a_resp[:,i],c='purple',label='respiration')
    # ax2.plot(t,a_mort[:,i],c='blue',label='mortality')
    # ax2.plot(t,a_diss[:,i],c='gray',label='dissolution')
    # ax2.plot(t,a_growth[:,i],c='green',label='growth')
    ax2.legend()
    
    ax[3].plot(t,a[:,i],c=colors[i],label=labels[i])
    # ax[3].plot(t,a_dead[:,i],c=colors[i+1],label=labels[i],linestyle='--')
    ax[3].legend()
    ax[3].set_title('attached living')
    ax2 = ax[3].twinx()
    # ax2.plot(t,a_resp[:,i],c='purple',label='respiration')
    # ax2.plot(t,a_mort[:,i],c='blue',label='mortality')
    # ax2.plot(t,a_coll[:,i],c='gray',label='collision')
    # ax2.plot(t,a_growth[:,i],c='green',label='growth')
    ax2.legend()
    
    
# plt.savefig('sensitivity_analysis.pdf')
  #%%
'''
fig,ax = plt.subplots(figsize=(18,18))
lines = ax.plot(t,zplt[:,:])
# Set line style
colors=['black','purple','orange','blue','green']
for i in range(len(a_diss_rate)):
    plt.setp(lines[i], color=colors[i],label=a_diss_rate[i])
ax.legend()
ax.grid()


fig,ax = plt.subplots(3,1,figsize=(18,18))
# Depth
ax[0].plot(t,zplt,label='No death')
ax[0].plot(t,zplt_2d,label='With death')
ax[0].grid(which='major')
ax[0].set_ylim([-70,2])
ax[0].set_ylabel('depth [m]',fontsize=14)
ax[0].legend(loc='lower left')
ax[0].set_title('1D verticle trajectories',fontsize=16)
ax[0].set_xlabel('time [d]',fontsize=14)
ax[0].set_xticks(tplt, labels=tticks)

# No death
ax[1].plot(t,a_resp,c='purple',label='respiration')
ax[1].plot(t,a_mort,c='blue',label='mortality')
ax[1].plot(t,a_coll,c='orange',label='collision')
ax[1].plot(t,a_growth,c='green',label='growth')
ax[1].plot(t,a_dead,c='k',label='attached dead')
ax[1].plot(t,a_diss,c='gray',label='dissolution')
ax[1].set_ylabel('terms [no.]',fontsize=14)
ax[1].legend(loc='upper left')
ax[1].set_ylim([-0.2e7,13e7])
ax[1].grid()
ax[1].set_xticks(tplt, labels=tticks)
ax[1].set_title('Contributions of different processes - No dead',fontsize=16)
ax[1].set_xlabel('time [d]',fontsize=14)
ax2=ax[1].twinx()
ax2.plot(t,a,c='darkred',label='attached living')
ax2.set_ylim([-0.2e10,8e10])


# With death
ax[2].plot(t,a_resp_2d,c='purple',label='respiration')
ax[2].plot(t,a_mort_2d,c='blue',label='mortality')
ax[2].plot(t,a_coll_2d,c='orange',label='collision')
ax[2].plot(t,a_growth_2d,c='green',label='growth')
ax[2].plot(t,a_dead_2d,c='k',label='attached dead')
ax[2].plot(t,a_diss_2d,c='gray',label='dissolution, rate='+str(a_diss_rate))
ax[2].set_ylabel('terms [no.]',fontsize=14)
ax[2].legend(loc='upper left')
ax[2].set_ylim([-0.2e7,10e7])
ax[2].grid()
ax[2].set_xticks(tplt, labels=tticks)
ax[2].set_title('Contributions of different processes - With dead',fontsize=16)
ax[2].set_xlabel('time [d]',fontsize=14)
ax2=ax[2].twinx()
ax2.plot(t,a_2d,c='darkred',label='attached living')
ax2.set_ylim([-0.2e10,8e10])

# plt.savefig(path+'2dn2d.png')
# plt.show()
'''