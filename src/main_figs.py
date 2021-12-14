import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mp
import matplotlib.patches as patches
import scipy.stats as st
from helpers import *
from collections import Counter

# fonts for figures
mp.rcParams.update({'font.size': 14})
mp.rcParams['axes.ymargin'] = 0.1
fs = 12

########################################################
# read data
########################################################

# load master experimental dataset
main_df = get_master_dataframe()
tids = main_df.index.unique() # unique ids
neworder = [3,4,6,7,10,11,15,27,0,1,2,5,8,9,12,13,14,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31]
tids = [tids[i] for i in neworder]

# sort experimental data into a single dictionary
datasets = {}
error = 0.2
for tid in tids:
    df = main_df.loc[tid].copy()
    datasets[tid] = df

# load posteriors into a dictionary
posteriors = {}
i = 1
print('load data')
for tid in tids:
    print(i,tid)
    i = i+1
    f = '../data/output/posteriors/separate/'+tid+'.csv'
    n = 100
    num_lines = sum(1 for l in open(f))
    skip_idx = [x for x in range(1, num_lines) if x % n != 0]
    posteriors[tid] = pd.read_csv(f,low_memory=False,skiprows=skip_idx)

########################################################
# load priors
########################################################

mu_prior,phi_prior,beta_prior,tau_prior,H0_prior,V0_prior = load_priors(df)

########################################################
# setup figures
########################################################

f2,ax2 = py.subplots(3,1,figsize=[12,12])
f3a,ax3a = py.subplots(4,4,figsize=[18,18])
f3b,ax3b = py.subplots(4,4,figsize=[18,18])
f3c,ax3c = py.subplots(4,4,figsize=[18,18])
f3d,ax3d = py.subplots(4,4,figsize=[18,18])
f4,ax4 = py.subplots()
f5,ax5 = py.subplots(4,2,figsize=[12,18])

ls = [0.1,0.28,0.57,0.75]
bs = [0.75,0.55,0.35,0.15]
w,h = [0.18,0.14]

ax3all = np.concatenate((ax3a.flatten(),ax3b.flatten(),ax3c.flatten(),ax3d.flatten()))
hosts = ax3all[0::2]
viruses = ax3all[1::2]
labs = list('abcdefghijklmnop')
for axes in [ax3a,ax3b,ax3c,ax3d]:
    for (a,l) in zip(axes.flatten(),labs):
        a.text(0.05,0.92,l,transform=a.transAxes)
    for (row,l) in zip(axes.T,ls):
        for (ax,b) in zip(row,bs):
            ax.set_position([l,b,w,h])
            if ax in viruses:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()

########################################################
# do plotting
########################################################

# dynamics
i = 1
allbests = pd.DataFrame()
bestmodels = []
print(' ')
print('begin plotting')
lowquant,higquant = 0.25,0.75 # quantiles for errorbars
for (hax,vax,tid) in zip(hosts,viruses,tids):
    print(i,tid)
    i = i+1
    ddf = datasets[tid]
    models = get_models(ddf)
    hax.set_xlabel('Time (hours)')
    vax.set_xlabel('Time (hours)')
    hax.set_ylabel(r'Host density (ml$^{-1}$)')
    vax.set_ylabel(r'Virus density (ml$^{-1}$)')
    hdat =ddf[ddf.organism=='H']
    vdat =ddf[ddf.organism=='V']
    hax.plot(hdat.time,hdat.abundance,marker='o')
    vax.plot(vdat.time,vdat.abundance,marker='o')
    mdf = posteriors[tid] # posteriors
    mi = mdf.loc[mdf.chi==min(mdf.chi)].index[0]
    bestmodelstring = mdf.iloc[mi]['Unnamed: 0']
    bestmodels.append(bestmodelstring)
    bestmodel = models[bestmodelstring]
    bestchain = mdf.iloc[mi]['chain#']
    bestmodelposteriors = mdf[mdf['Unnamed: 0']==bestmodelstring]
    bestmodelposteriors = bestmodelposteriors[bestmodelposteriors['chain#']==bestchain]
    bestmodelposteriors['HostSpecies'] = hdat.HostSpecies.unique()[0]
    bestmodelposteriors['HostClass'] = hdat.HostClass.unique()[0]
    bestmodelposteriors['VirusName'] = vdat.VirusName.unique()[0]
    bestmodelposteriors['id'] = tid
    allbests = pd.concat((allbests,bestmodelposteriors))
    set_optimal_parameters(bestmodel,bestmodelposteriors)
    mod = bestmodel.integrate()
    hax.plot(bestmodel.times,mod['H'],c='r',lw=2,zorder=2)
    vax.plot(bestmodel.times,mod['V'],c='r',lw=2,zorder=2)
    hax.set_title(hdat.HostSpecies.unique()[0])
    vax.set_title(vdat.VirusName.unique()[0])
    for a in range(100):
        set_random_param(bestmodel,bestmodelposteriors)
        mod = bestmodel.integrate()
        hax.plot(bestmodel.times,mod['H'],c=str(0.8),lw=1,zorder=1)
        vax.plot(bestmodel.times,mod['V'],c=str(0.8),lw=1,zorder=1)
    hax.semilogy()
    vax.semilogy()

allbests.set_index('id')
allbests['lam'] = 1/allbests.tau

# write to csv file
allbests.to_csv('../data/output/posteriors/collated/collated_posteriors.csv')

# main boxplot of posteriors
for (a,param,l) in zip(ax2,['phi','lam','beta'],list('abc')):
    grouped = allbests.loc[:,['VirusName','HostClass', param]] \
        .groupby(['HostClass','VirusName']) \
        .median() \
        .sort_values(by=['HostClass',param]) 
    sns.boxplot(x='VirusName',y=param,data=allbests,\
            ax=a,order=grouped.index.get_level_values(1),\
            hue='HostClass',dodge=False)
    a.get_legend().remove()
    py.sca(a)
    a.text(0.07,0.9,l,transform=a.transAxes)
    py.xticks(rotation=45)
f2.subplots_adjust(hspace=0.4)
for a in ax2:
    a.semilogy()

# bar chart showing infection states
states = ['oneI','twoI','threeI','fourI','fiveI','sixI','sevenI','eightI','nineI','tenI']
counts = []
for s in states:
    counts.append(sum([b == s for b in bestmodels]))
ax4.bar(range(1,11), counts)
ax4.set_xlabel('Number of infection states')
ax4.set_ylabel('Number of datasets')

# sensitivity plots
dset = 'Baudoux33'
ddf = datasets[dset]
mdf = posteriors[dset]
mi = mdf.loc[mdf.chi==min(mdf.chi)].index[0]
basemodelstring = mdf.iloc[mi]['Unnamed: 0']
basemodelstring = 'oneI'
models = get_models(ddf)
lmod = models[basemodelstring]
lposteriors = mdf[mdf['Unnamed: 0']==basemodelstring]
set_optimal_parameters(lmod,lposteriors)
for ax in ax5:
    mod = lmod.integrate()
    ax[0].plot(mod.time,mod.H,c='k')
    ax[1].plot(mod.time,mod.V,c='k')
    ax[0].set_xlabel('Time (hours)')
    ax[1].set_xlabel('Time (hours)')
    ax[0].set_ylabel(r'Host density (ml$^{-1}$)')
    ax[1].set_ylabel(r'Virus density (ml$^{-1}$)')
    ax[0].semilogy()
    ax[1].semilogy()

default_set = lmod.get_parameters(as_dict=True)
default_inits = lmod.get_inits(as_dict=True)

phi_low = default_set.copy()
phi_high = default_set.copy()
phi_low['phi'] = default_set['phi']/2
phi_high['phi'] = default_set['phi']*2

tau_low = default_set.copy()
tau_high = default_set.copy()
tau_low['tau'] = default_set['tau']/1.2
tau_high['tau'] = default_set['tau']*1.2

beta_low = default_set.copy()
beta_high = default_set.copy()
beta_low['beta'] = default_set['beta']/2
beta_high['beta'] = default_set['beta']*2

# phi
lmod.set_parameters(**phi_low)
mod = lmod.integrate()
ax5[1][0].plot(mod.time,mod.H,c='b',ls='--',label=r'low $\phi$')
ax5[1][1].plot(mod.time,mod.V,c='b',ls='--',label=r'low $\phi$')
lmod.set_parameters(**phi_high)
mod = lmod.integrate()
ax5[1][0].plot(mod.time,mod.H,c='r',ls='-.',label=r'high $\phi$')
ax5[1][1].plot(mod.time,mod.V,c='r',ls='-.',label=r'high $\phi$')

# tau
lmod.set_parameters(**tau_low)
mod = lmod.integrate()
ax5[2][0].plot(mod.time,mod.H,c='b',ls='--',label=r'low $\tau$')
ax5[2][1].plot(mod.time,mod.V,c='b',ls='--',label=r'low $\tau$')
lmod.set_parameters(**tau_high)
mod = lmod.integrate()
ax5[2][0].plot(mod.time,mod.H,c='r',ls='-.',label=r'high $\tau$')
ax5[2][1].plot(mod.time,mod.V,c='r',ls='-.',label=r'high $\tau$')

# beta
lmod.set_parameters(**beta_low)
mod = lmod.integrate()
ax5[3][0].plot(mod.time,mod.H,c='b',ls='--',label=r'low $\beta$')
ax5[3][1].plot(mod.time,mod.V,c='b',ls='--',label=r'low $\beta$')
lmod.set_parameters(**beta_high)
mod = lmod.integrate()
ax5[3][0].plot(mod.time,mod.H,c='r',ls='-.',label=r'high $\beta$')
ax5[3][1].plot(mod.time,mod.V,c='r',ls='-.',label=r'high $\beta$')

# nstates
basemodelstring = 'twoI'
lmod = models[basemodelstring]
lmod.set_parameters(**default_set)
lmod.set_inits(**default_inits)
mod = lmod.integrate()
ax5[0][0].plot(mod.time,mod.H,c='b',ls='--',label=r'$n=2$')
ax5[0][1].plot(mod.time,mod.V,c='b',ls='--',label=r'$n=2$')

basemodelstring = 'threeI'
lmod = models[basemodelstring]
lmod.set_parameters(**default_set)
lmod.set_inits(**default_inits)
mod = lmod.integrate()
ax5[0][0].plot(mod.time,mod.H,c='r',ls='-.',label=r'$n=3$')
ax5[0][1].plot(mod.time,mod.V,c='r',ls='-.',label=r'$n=3$')

style = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowstyle=style, color="k")

a1 = patches.FancyArrowPatch((0, 1.1e+7), (20, 1.1e+7), **kw)
a2 = patches.FancyArrowPatch((10,2.2e+6), (50,2.2e+6),connectionstyle="arc3,rad=-.25", **kw)
a3 = patches.FancyArrowPatch((35, 8e+5), (48, 3.5e+5), **kw)
a4 = patches.FancyArrowPatch((60, 2e+7), (60, 2e+8), **kw)

t1 = 'lysis delay'
t2 = 'timing of host demise'
t3 = 'gradient of\nhost demise'
t4 = 'virus yield'

tx = (0,10,15,40)
ty = (6e+6,4e+6,6e+5,5e+7)
texts = (t1,t2,t3,t4)
patches = [a1,a2,a3,a4]
paxes = [ax5[0][1],ax5[1][0],ax5[2][0],ax5[3][1]]

for (ax,patch,x,y,t) in zip(paxes,patches,tx,ty,texts):
    ax.add_patch(patch)
    ax.text(x,y,t)

paxes[0].set_ylim(ymin=4e+6)
paxes[1].set_ylim(ymax=6e+6)

labs = ['a','b','c','d','e','f','g','h']
for (ax,l) in zip(ax5.flatten(),labs):
    ax.text(0.05,0.9,l,transform=ax.transAxes)

for ax in ax5:
    l = ax[0].legend()
    l.draw_frame(False)

f5.subplots_adjust(hspace=0.3)

########################################################
# save figures
########################################################

f2.savefig('../figures/posterior_boxplot',bbox_inches='tight',pad_inches=0.1)
f3a.savefig('../figures/dynamics_a',bbox_inches='tight',pad_inches=0.1)
f3b.savefig('../figures/dynamics_b',bbox_inches='tight',pad_inches=0.1)
f3c.savefig('../figures/dynamics_c',bbox_inches='tight',pad_inches=0.1)
f3d.savefig('../figures/dynamics_d',bbox_inches='tight',pad_inches=0.1)
f4.savefig('../figures/infection_states',bbox_inches='tight',pad_inches=0.1)
f5.savefig('../figures/sensitivity',bbox_inches='tight',pad_inches=0.1)
