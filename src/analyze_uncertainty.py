import scipy
import pylab as pl
import numpy as np
from helpers import *
from matplotlib.backends.backend_pdf import PdfPages
import ODElib

# fonts for figures
fs = 12

# read data
master_df = pd.read_csv('../data/input/rawdata/with_reps.csv')
metadata = pd.read_csv('../data/input/metadata/metamaster.csv',index_col='id')
master_df = master_df.merge(metadata,on='id')

master_df['meanraw'] = np.mean(np.r_[[master_df[i] for i in ['rep1','rep2','rep3']]],axis=0)
master_df['stdraw'] = np.std(np.r_[[master_df[i] for i in ['rep1','rep2','rep3']]],axis=0)
master_df['meanlog'] = np.mean(np.log(np.r_[[master_df[i] for i in ['rep1','rep2','rep3']]]),axis=0)
master_df['stdlog'] = np.std(np.log(np.r_[[master_df[i] for i in ['rep1','rep2','rep3']]]),axis=0)

master_df = master_df.set_index('organism')

treatments = master_df.query('control==False').copy() # remove controls
controls = master_df.query('control==True').copy() # remove controls

controls = controls.mask(controls.rep1==0)
controls = controls.mask(controls.rep2==0)
controls = controls.mask(controls.rep3==0)

hosttrets = treatments.loc['H']
virtrets = treatments.loc['V']

# first load subset of data that contain replicate
ndfh,ndfv = master_df.loc['H'],master_df.loc['V']
ndfh = ndfh[ndfh['VirusName']=='Ehv99B1']
ndfv = ndfv[ndfv['VirusName']=='Ehv99B1']

fa,axa = py.subplots(1,2,figsize=[9,4])
f,ax = pl.subplots(2,2,figsize=[10,8])
f1,ax1 = pl.subplots(1,2,figsize=[9,4.5])

# raw plots
axa[0].plot(ndfh.time,ndfh.rep1,'-o',label='rep 1')
axa[0].plot(ndfh.time,ndfh.rep2,'-^',label='rep 2')
axa[0].plot(ndfh.time,ndfh.rep3,'-*',label='rep 3')
axa[1].plot(ndfv.time,ndfv.rep1,'-o')
axa[1].plot(ndfv.time,ndfv.rep2,'-^')
axa[1].plot(ndfv.time,ndfv.rep3,'-*')
l = axa[0].legend(prop={'size':fs})
l.draw_frame(False)
axa[0].semilogy()
axa[1].semilogy()
for a in axa:
        a.set_xlabel('Time (hours)',fontsize=fs)
axa[0].set_ylabel('Host (ml$^{-1}$)',fontsize=fs)
axa[1].set_ylabel('Virus (ml$^{-1}$)',fontsize=fs)
fa.subplots_adjust(wspace=0.3)

# histograms
nbins = 20
hbins = np.logspace(np.log(np.amin(hosttrets.stdlog)),np.log(np.amax(hosttrets.stdlog)),20)
vbins = np.logspace(np.log(np.amin(virtrets.stdlog)),np.log(np.amax(virtrets.stdlog)),20)
ax1[0].hist(hosttrets.stdlog,bins=hbins)
ax1[1].hist(virtrets.stdlog,bins=vbins)
for a in ax1:
    a.semilogx()
    a.axvline(x=0.2,c='g')
ax1[0].set_xlim([1e-2,2])
ax1[1].set_xlim([2e-2,3])
ax1[0].set_xlabel('Host abundance log-transformed variance')
ax1[1].set_xlabel('Virus abundance log-transformed variance')
ax1[0].set_ylabel('Frequency')
ax1[1].set_ylabel('Frequency')

# view error vs. mean - hosts
gcor,dcor = np.r_[[]],np.r_[[]]
glcor,dlcor = np.r_[[]],np.r_[[]]
for t in hosttrets.VirusName.unique():
    df = hosttrets[hosttrets['VirusName']==t]
    cor = df.corr(method='pearson')
    rawcor = cor.loc['meanraw'].stdraw
    logcor = cor.loc['meanlog'].stdlog
    dcor = np.append(dcor,rawcor)
    dlcor = np.append(dlcor,logcor)
    df.plot(x='meanraw',y='stdraw',ax=ax[0,0],loglog=True,marker='o',label=t,legend=False)
    df.plot(x='meanlog',y='stdlog',ax=ax[1,0],label=t,legend=False,marker='o',ylim=[-0.2,2])
l = ax[0,0].legend(loc='lower right')
l.draw_frame(False)

# view error vs. mean - viruses
for t in virtrets.VirusName.unique():
    df = virtrets[virtrets['VirusName']==t]
    cor = df.corr(method='pearson')
    rawcor = cor.loc['meanraw'].stdraw
    logcor = cor.loc['meanlog'].stdlog
    gcor = np.append(gcor,rawcor)
    glcor = np.append(glcor,logcor)
    df.plot(x='meanraw',y='stdraw',ax=ax[0,1],loglog=True,marker='o',label=t,legend=False)
    df.plot(x='meanlog',y='stdlog',ax=ax[1,1],marker='o',label=t,legend=False,ylim=[-0.2,2])
ax[0,0].set_xlabel('mean of raw abundance')
ax[0,1].set_xlabel('mean of raw abundance')
ax[1,0].set_xlabel('mean of logged abundance')
ax[1,1].set_xlabel('mean of logged abundance')
ax[0,0].set_ylabel('standard deviation of raw abundance')
ax[0,1].set_ylabel('standard deviation of raw abundance')
ax[1,0].set_ylabel('standard deviation of logged abundance')
ax[1,1].set_ylabel('standard deviation of logged abundance')
ax[0,0].set_title('hosts')
ax[0,1].set_title('viruses')

for (a,l) in zip(ax.flatten(),'abcd'):
    a.text(0.05,0.9,l,transform=a.transAxes)
for (a,l) in zip(ax1.flatten(),'abcd'):
    a.text(0.05,0.9,l,transform=a.transAxes)

# correlation analysis
f2,ax2 = py.subplots()
ax2.hist(np.append(gcor,dcor),alpha=0.5,label='unlogged data')
ax2.hist(np.append(glcor,dlcor),alpha=0.5,label='logged data')
l = ax2.legend(ncol=2)
l.draw_frame(False)
ax2.set_xlabel('Pearsons r correlation between variance and mean')
ax2.set_ylabel('Frequency')
ax2.axvline(x=0,ls='--',c='k')
ax2.set_ylim([0,6])

f.subplots_adjust(hspace=0.3,wspace=0.3)

fa.savefig('../figures/nissimov_raw',bbox_inches='tight',pad_inches=0.1)
f.savefig('../figures/nissimov_correlation',bbox_inches='tight',pad_inches=0.1)
f1.savefig('../figures/nissimov_hists',bbox_inches='tight',pad_inches=0.1)
f2.savefig('../figures/nissimov_pearson',bbox_inches='tight',pad_inches=0.1)

pl.show()
