import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mp
import matplotlib.patches as patches
import scipy.stats as st
import scipy as sp
from helpers import *
from collections import Counter

#####################################################
# setup figures
#####################################################

f1,ax1 = py.subplots(2,2,figsize=[10,10])
f2,ax2 = py.subplots(2,2,figsize=[10,10])
ax1 = ax1.flatten()
ax2 = ax2.flatten()

# fonts for figures
mp.rcParams.update({'font.size': 14})
mp.rcParams['axes.ymargin'] = 0.1
fs = 12

#####################################################
# read in size data
#####################################################

metadat = pd.read_csv('../data/input/metadata/metamaster.csv').set_index('id')
params = pd.read_csv('../data/output/posteriors/collated/collated_posteriors.csv').set_index('id')

metadat = metadat.drop('Nissimov2020b').copy()
metadat = metadat.drop('Nissimov2020c').copy()

hr = (metadat[metadat.columns[8]]+metadat[metadat.columns[9]])/2.0
vr = (metadat[metadat.columns[10]]+metadat[metadat.columns[10]])*0.5/1000 

vsize = 4/3.0*np.pi*(vr**3)
hsize = 4/3.0*np.pi*(hr**3)

#####################################################
# read in model fits
#####################################################

betas,phis,groups = np.r_[[]],np.r_[[]],[]
for tid in metadat.index:
    betas = np.append(betas,np.median(params.loc[tid].beta))
    phis = np.append(phis,np.median(params.loc[tid].phi))
    groups.append(metadat.loc[tid]['HostClass'])
    print('tid',tid,metadat.loc[tid]['HostClass'])
epsilons = betas*vsize/hsize
phispec = phis/vsize
rhos = enc_brown(hr,vr,upreyswim=False,Dpreydiff=False) / 1e+12 * 86400.0
rhospec = rhos/vsize
mdic = {'beta':betas,'phi':phis,'Host class':groups,'vsize':vsize,'hsize':hsize,\
        'vradius':vr,'hradius':hr,'epsilon':epsilons,'phispec':phispec,\
        'rhos':rhos,'rhospec':rhospec,'Infection efficiency':phis/rhos}
simpdat = pd.DataFrame(mdic)
simpdat = simpdat.sort_values('vradius')

#####################################################
# infection affinity / transfer efficiency vs. size
#####################################################

# infection affinity
sns.scatterplot('vradius','phi',hue='Host class',data=simpdat,ax=ax1[0],legend=False)

# burst size
sns.scatterplot('vradius','beta',hue='Host class',data=simpdat,ax=ax1[1],legend=False)

# normalized infection affinity
sns.scatterplot('vradius','phispec',hue='Host class',data=simpdat,ax=ax1[2],legend=False)

# transfer efficiency
sns.scatterplot('vradius','epsilon',hue='Host class',data=simpdat,ax=ax1[3])
ax1[3].legend(fontsize='10')

#####################################################
# adsorption efficiency
#####################################################

# contact rate vs. infection rate
sns.scatterplot('rhos','phi',hue='Host class',data=simpdat,ax=ax2[0],legend=False)
myrange = np.linspace(min(simpdat.phi),max(simpdat.rhos),1000)
ax2[0].plot(myrange,myrange,c='k',ls='--')
ax2[0].semilogx()
ax2[0].semilogy()

# vs. size
sns.scatterplot('vradius','Infection efficiency',hue='Host class',data=simpdat,ax=ax2[1],legend=False)
ax2[1].axhline(y=1.0,c='k',ls='--')
ax2[1].semilogy()

# raw histograms
phibins = np.logspace(min(np.log10(simpdat.phi)),max(np.log10(simpdat.phi)),10)
rhobins = np.logspace(min(np.log10(simpdat.rhos)),max(np.log10(simpdat.rhos)),10)
ax2[2].hist(simpdat.phi,bins=phibins,label='Infection affinity')
ax2[2].hist(simpdat.rhos,bins=rhobins,label='Encounter rate')
l = ax2[2].legend()
l.draw_frame(False)
ax2[2].semilogx()

# infection efficiency histogram
effbins = np.logspace(min(np.log10(simpdat['Infection efficiency'])),max(np.log10(simpdat['Infection efficiency'])),10)
ax2[3].hist(simpdat['Infection efficiency'],bins=effbins)
ax2[3].axvline(x=1.0,ls='--',c='k',label='Theoretical upper limit')
ax2[3].axvline(x=np.median(simpdat['Infection efficiency']),c='r',lw=1.5,label='Median infection efficiency')
l = ax2[3].legend()
l.draw_frame(False)
ax2[3].semilogx()

#####################################################
# final tweaks
#####################################################

for a in ax1:
    a.set_xlabel('Virus radius ($\mu$m)')
ax1[0].set_ylabel('Infection affinity, $\phi$ (ml day$^{-1}$)')
ax1[1].set_ylabel(r'Burst size, $\beta$ (ml day$^{-1}$)')
ax1[2].set_ylabel(r'Specific infection affinity, $\frac{\phi}{Q_v}$ (day$^{-1}$)')
ax1[3].set_ylabel('Transfer efficiency, $\epsilon$ (-)')
for a in ax1:
    a.semilogy()

ax2[0].set_ylabel('Infection affinity, $\phi$ (ml day$^{-1}$)')
ax2[1].set_ylabel(r'Infection efficiency, $\xi$ (-)')
ax2[2].set_ylabel('Frequency')
ax2[3].set_ylabel('Frequency')

ax2[0].set_xlabel(r'Theoretical encounter, $\rho$ (ml day$^{-1}$)')
ax2[1].set_xlabel(r'Virus radius ($\mu$m)')
ax2[2].set_xlabel(r'Infection or encounter rate (ml day$^{-1}$)')
ax2[3].set_xlabel('Infection efficiency, $\epsilon$ (-)')

ax2[2].set_ylim([0,12])
ax2[3].set_ylim([0,10])

for (a,b,l) in zip(ax1,ax2,list('abcd')):
    a.text(0.07,0.9,l,transform=a.transAxes)
    b.text(0.07,0.9,l,transform=b.transAxes)

f1.subplots_adjust(hspace=0.3,wspace=0.3)
f2.subplots_adjust(hspace=0.3,wspace=0.3)

f1.savefig('../figures/derived_traits',bbox_inches='tight',pad_inches=0.1)
f2.savefig('../figures/infection_efficiency',bbox_inches='tight',pad_inches=0.1)

simpdat.to_csv('../data/output/master/master_param_file.csv')

py.show()



