'''#########################################################################
            Blind Source Separation Denoising (BSSD) Functions
   #########################################################################'''

def get_noise_power(ROIs_means_corrected,time,inactive_msk=None):
    import numpy as np
    nregions = ROIs_means_corrected.shape[1]
    if np.any(inactive_msk)==None:
        inactive_msk = np.ones_like(time,dtype=bool)
    noise = np.zeros((nregions,))
    for j in range(nregions):
        #Get NOISE POWER
        noise[j] = np.sum(np.square(abs(ROIs_means_corrected[inactive_msk,j])),axis=0)/(ROIs_means_corrected[inactive_msk,:].shape[0])
    return(noise)

def guess_t_sig_active_time(sources, t_sig_onset_idx):
    '''Returns indices of time vector later then t_sig_onset and when they are bigger than channel background noise'''
    import numpy as np
    noise_threshold = np.mean(sources[:t_sig_onset_idx,:],axis=0) + 2*np.std(sources[:t_sig_onset_idx,:],axis=0)
    t_sig_active = np.empty_like(sources,dtype=bool)
    for i in range(len(noise_threshold)):
        t_sig_active[:,i] = np.where(sources[:,i]>noise_threshold[i],True,False)
        t_sig_active[:t_sig_onset_idx,i] = False
    return(noise_threshold, t_sig_active)

def get_signal_power(ROIs_means_corrected,time,inactive_msk=None):
    import numpy as np
    nregions = ROIs_means_corrected.shape[1]
    if np.any(inactive_msk)==None:
        inactive_msk = np.ones_like(time,dtype=bool)
    active_mask = np.invert(inactive_msk)
    #Get SIGNAL POWER
    signal_power = np.sum(np.square(abs(ROIs_means_corrected[active_mask])),axis=0)/(len(active_mask))
    return(signal_power)

def manual_select():
    import numpy as np
    nb = input('Enter one or more sources number (separate numbers by "," if number of sources > 1): ')
    selected_source_idx = []
    if nb.find(',')==-1: #Single source selected
        try:
            number = int(nb)
            selected_source_idx.append(number)
        except ValueError:
            print("Invalid number")
    else:
        nb = nb.replace(" ","")
        commas_idx = []
        import re
        for m in re.finditer(',', nb):
            commas_idx.append(m.start())
        try:
            number = int(nb[:commas_idx[0]])
            selected_source_idx.append(number)
            for i in range(len(commas_idx)):
                if i==len(commas_idx)-1:
                    number = int(nb[commas_idx[i]+1:])
                else:
                    number = int(nb[commas_idx[i]+1:commas_idx[i+1]])
                selected_source_idx.append(number)
        except ValueError:
            print("Invalid number list")
    return(np.array(selected_source_idx))

def wavelet_denoise(S, time, wave='dmey'):
    import numpy as np
    import pywt
    from statsmodels.robust.scale import mad
    w = pywt.Wavelet(wave)
    shape = S.shape
    n = shape[0]
    #Denoise just one component
    if len(shape)==1:
        if (n%2!=0):
            Y = np.zeros((n+1,))
        else:
            Y = np.zeros_like(S)
        max_level = pywt.dwt_max_level(n, w)
        coeff = pywt.wavedec(S, w, mode='periodization',level=max_level)

        sigma = mad(coeff[-1])
        K = np.sqrt(2*np.log(time.shape[0]))*sigma

        coeff_T = []
        for j in range(len(coeff)):
            c = pywt.threshold(coeff[j], K, 'hard')
            coeff_T.append(c)
        Y = pywt.waverec(coeff_T, w ,mode='periodization')
        if (n%2!=0):
            Y = Y[:len(Y)-1]
    return(Y)

def auto_select_signal(sources, time, t_sig_onset_idx):
    import numpy as np
    noise_threshold = 2*np.std(sources[:t_sig_onset_idx,:],axis=0)
    t_sig_active = np.where(abs(sources-np.mean(sources[:t_sig_onset_idx,:],axis=0))>noise_threshold,True,False)
    t_sig_active[:t_sig_onset_idx] = False
    sources_abovenoise_power = np.empty((sources.shape[1],))
    for i in range(sources.shape[1]):
        active_idxs = get_longest_ones_seq_idx(t_sig_active[:,i])
        if np.any(active_idxs)==None:
            sources_abovenoise_power[i] = 0
        else:
            sources_abovenoise_power[i] = np.sum(np.square(sources[active_idxs,i]))
    sources_idx_sorted = np.argsort(sources_abovenoise_power)
    sources_idx_sorted_inverse = sources_idx_sorted[::-1] #sorts from max to min
    return(sources_idx_sorted_inverse)

def get_longest_ones_seq_idx(t_sig_active):
    import numpy as np
    ones_pos = np.nonzero(t_sig_active)[0]
    if ones_pos.size==0:
        return(None)
    else:
        d_ones_pos = np.diff(ones_pos)
        d_ones_pos = np.insert(d_ones_pos,len(d_ones_pos),0)
        seq=0
        maxseq=0
        for i in range(len(ones_pos)):
            if d_ones_pos[i]==1:
                seq+=1
            else:
                if ((seq+1)>maxseq):
                    maxseq = seq+1
                    pos = ones_pos[i] - seq
                    pos_end = ones_pos[i]+1
                seq = 0
        longest_idxs = np.arange(pos,pos_end)
    return(longest_idxs)

'''#########################################################################
                            BSSD Main Function
   #########################################################################'''

def denoise(ROIs_means_corrected,time,inactive_msk,t_sig_onset,
            method: str='ICA',n_comp: int=2,wavelet: str='Haar',whiten: bool=True,
            autoselect: str='auto'):
    import numpy as np
    import matplotlib.pyplot as plt
    # autoselect = 'auto'
    if t_sig_onset==0:
        t_sig_onset = None
    if n_comp==1:
        n_comp=2
        autoselect='auto'

    noise_power = get_noise_power(ROIs_means_corrected,time,inactive_msk=inactive_msk)

    if (method=='ICA') | (method=='wICA'):
        from sklearn.decomposition import FastICA
        bss = FastICA(n_components=n_comp,max_iter=2000,tol=0.01)
    if (method=='PCA') | (method=='wPCA'):
        from sklearn.decomposition import PCA
        bss = PCA(n_components=n_comp,whiten=whiten)
    t_sig_onset_idx = np.argmin(abs(time-t_sig_onset))

    components = bss.fit_transform(ROIs_means_corrected)  # Estimate sources
    components_filt = np.zeros_like(components)
    sources_idx_sorted_inverse = auto_select_signal(components, time, t_sig_onset_idx)
    selected_source_idx = sources_idx_sorted_inverse[0]

    components_filt[:,selected_source_idx] = components[:,selected_source_idx]
    selected_source_idx = np.atleast_1d(selected_source_idx)

    if (method=='wPCA') | (method=='wICA'):
        for i in range(np.size(selected_source_idx)):
            components_filt[:,selected_source_idx[i]] = wavelet_denoise(components_filt[:,selected_source_idx[i]],time,wave=wavelet)
    ROIs_means_filtered = bss.inverse_transform(components_filt)
    ROIs_means_filtered = ROIs_means_filtered - np.median(ROIs_means_filtered[inactive_msk,:],axis=0)

    signal_power = get_signal_power(ROIs_means_filtered,time,inactive_msk)

    SNR = signal_power/noise_power
    SNR_dB = np.zeros_like(SNR)
    SNR_dB[SNR>0] = 10*np.log10(SNR[SNR>0])

    return(ROIs_means_filtered,components,selected_source_idx,SNR_dB)

