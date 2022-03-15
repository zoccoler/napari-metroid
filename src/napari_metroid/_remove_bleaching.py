'''#########################################################################
                        Remove Photobleaching Functions
   #########################################################################'''
from napari.types import ImageData, LabelsData
def guess_t_sig_onset(vmean,time):
    '''Returns time and correspondig index where absolute maximum derivative is found, indicating signal rise/fall time'''
    import numpy as np
    d_vmean = abs(np.diff(vmean))
    dmax_idx = np.argmax(d_vmean) +1
    t_sig_onset = time[dmax_idx]
    return(t_sig_onset, dmax_idx)

def guess_t_sig_duration(vmean, time, t_sig_onset_idx):
    '''Returns indices of time vector later then t_sig_onset and when they are bigger than channel background noise'''
    import numpy as np
    noise_threshold = 2*np.std(vmean[:t_sig_onset_idx])
    t_sig_active = np.empty_like(vmean,dtype=bool)
    t_sig_active = np.where(abs(vmean)>noise_threshold,True,False)
    t_sig_active[:t_sig_onset_idx] = False
    i=t_sig_onset_idx
    while(t_sig_active[i]==True):
        i+=1
    t_sig_end_idx = i
    return(time[t_sig_end_idx],t_sig_end_idx)

def guess_t_sig_prop(video,time,mask,transitory,t_sig_onset=None):
    '''Tries to find where signal starts and its duration by cell mask mean over time'''
    from scipy.signal import detrend
    import numpy as np
    video_masked = video*mask
    video_masked_mean = np.mean(video_masked,axis=(1,2))

    video_masked_mean_detrend = detrend(video_masked_mean,type='linear')
    if t_sig_onset is None:
        t_sig_onset, t_sig_onset_idx = guess_t_sig_onset(video_masked_mean_detrend,time)
    else:
        t_sig_onset_idx = np.argmin(abs(time-t_sig_onset))

    if transitory==True:
        t_sig_end,t_sig_end_idx = guess_t_sig_duration(video_masked_mean_detrend, time, t_sig_onset_idx)

        if t_sig_onset==t_sig_end:
            t_sig_end=None
        return(t_sig_onset, t_sig_onset_idx, t_sig_end)
    else:
        return(t_sig_onset, t_sig_onset_idx,None)

def create_inactive_idx_msk(video,time,mask,transitory=False,t_sig_onset=None,t_sig_end=None):
    '''Create vector masks indicating time periods of inactivity, i.e., where only noise is present'''
    import numpy as np
    #If there is no information about signal start, try to estimate from mean from cellmasked video
    if transitory==None:
        idx_msk = np.ones_like(time,dtype=bool)
        return(idx_msk,t_sig_onset,t_sig_end)

    if transitory==False:
        t_sig_end=None
        if (t_sig_onset==None):
            t_sig_onset,t_sig_onset_idx,t_sig_end = guess_t_sig_prop(video, time, mask,transitory)

    if transitory==True:
        if (t_sig_onset==None):
            t_sig_onset,t_sig_onset_idx,t_sig_end = guess_t_sig_prop(video, time, mask,transitory)
        elif t_sig_end is None:
            t_sig_onset,t_sig_onset_idx,t_sig_end = guess_t_sig_prop(video, time, mask,transitory,t_sig_onset)
            if t_sig_onset==t_sig_end:
                t_sig_end = None

    if t_sig_end is None:
        idx_msk = ((time>0) & (time<t_sig_onset-0.1))
    else:
        idx_msk = ((time>0) & (time<t_sig_onset-0.1)) | ((time>t_sig_end))

    return(idx_msk,t_sig_onset,t_sig_end)

def monoexp(x, a, b, d):
    import numpy as np
    return a * np.exp(-b * x) + d
def monoexp_and_line(x, a, b, c, d):
    import numpy as np
    return a * np.exp(-b * x) - (c * x) + d
def monoexp_line_step(x, a, b, c, d, e,t_sig_onset):
    import numpy as np
    return a * np.exp(-b * x) - (c * x) + d - e * (np.sign(x-t_sig_onset) + 1)
def monoexp_step(x, a, b, d, e,t_sig_onset):
    import numpy as np
    return a * np.exp(-b * x) + d - e * (np.sign(x-t_sig_onset) + 1)

def photob_fit(ROIs_means,time,idx_msk,transitory,t_sig_onset):
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.stats import linregress
    steps = np.empty(ROIs_means.shape[1])
    corrected = np.empty(ROIs_means.shape)
    #Runs over the ROIs
    for j in range(ROIs_means.shape[1]):
        #Photobleaching is taken from inactivity intervals
        photobleaching = ROIs_means[idx_msk,j]

        #Tries different fit functions:
        #  1. Linear (just to get an estimative for curve_fit parameters)
        #  2. Monoexponential + linear for AP signals
        #  3. Monoexponential + linear + step (just for step-like signal)

        y0 = np.mean(photobleaching[0:5])
        yf = np.mean(photobleaching[-5:])
        #FIT #1:    linear fit
        c, d, r_value, p_value, std_err= linregress(time[idx_msk],photobleaching)
#         c, d = np.polyfit(time[idx_msk],photobleaching,1)
        p = np.poly1d([c,d])
        #Calculate residues
        res_linear = photobleaching - p(time[idx_msk])
        total_res_linear = np.sum(abs(res_linear**2))/len(photobleaching)

        #Assures y0 is bigger than yf for proper upper boundaries calculation
        if yf>=y0:
            if d>yf:
                y0 = d
            else:
                y0 = yf + 1

        #If transitory signal
        if ((transitory==True) | (transitory==None)):
            #FIT #2:    monoexp and line fit
            upper_bounds = [2*(y0-yf), 2*(np.log(d-(c*yf)-c)-np.log(y0-yf)), 2*(abs(c)), 2*yf]
            popt2, pcov2 = curve_fit(monoexp_and_line, time[idx_msk], photobleaching, bounds=(0, upper_bounds))
            res_expline = photobleaching - monoexp_and_line(time[idx_msk], *popt2)
            total_res_expline = np.sum(abs(res_expline**2))/len(photobleaching)

            #If any fit parameter gets close to its respective upper_boundary, expand upper_boundaries and retries fit
            if np.all(np.greater(upper_bounds,popt2+0.01*(popt2)))==False:
                clip_flag = 1
                while(clip_flag==1):
                    close = np.isclose(upper_bounds,popt2,atol=1e-02)
                    if np.any(close)==False:  #if can't identify which parameter is clipped, duplicate all
                        upper_bounds = np.multiply(2,upper_bounds)
                    else:                     #if clipped parameter is identified, duplicate it
                        closeidx = np.argwhere(close)[0][0]
                        upper_bounds[closeidx] = np.multiply(2,upper_bounds[closeidx])
                    popt2, pcov2 = curve_fit(monoexp_and_line, time[idx_msk], photobleaching, bounds=(0, upper_bounds))
                    res_expline = photobleaching - monoexp_and_line(time[idx_msk], *popt2)
                    total_res_expline = np.sum(abs(res_expline**2))/len(photobleaching)
                    if np.all(np.greater(upper_bounds,popt2+0.01*(popt2)))==True:
                        clip_flag = 0
                    #Breaks if either exponential coefficient or linear coefficients "overflow"
                    if (popt2[0]+popt2[3]>1.2*y0)|(popt2[1]<0.05):
                        break
        #If step-like signal, perform fits including step
        #Inputs include whole data
        if transitory==False:
            #FIT #5:     monoexp, line and step fit
            upper_bounds = [2*(y0-yf), 2*(np.log(d-(c*yf)-c)-np.log(y0-yf)), 2*(abs(c)), 2*yf, (y0-yf)]
            popt3, pcov3 = curve_fit(lambda x, a, b, c, d, e: monoexp_line_step(x,a,b,c,d,e,t_sig_onset), time, ROIs_means[:,j], bounds=(0, upper_bounds))
            res_explinestep = ROIs_means[:,j] - monoexp_line_step(time, *np.insert(popt3,len(popt3),t_sig_onset))
            total_res_explinestep = np.sum(abs(res_explinestep**2))/len(ROIs_means[:,j])
            #If any fit parameter gets close to its respective upper_boundary, expand upper_boundaries and retries fit
            if np.all(np.greater(upper_bounds,popt3+0.01*(popt3)))==False:
                clip_flag = 1
                while(clip_flag==1):
                    close = np.isclose(upper_bounds,popt3,atol=1e-02)
                    if np.any(close)==False:  #if can't identify which parameter is clipped, duplicate all
                        upper_bounds = np.multiply(2,upper_bounds)
                    else:
                        closeidx = np.argwhere(close)[0][0]
                        upper_bounds[closeidx] = np.multiply(2,upper_bounds[closeidx])
                    popt3, pcov3 = curve_fit(lambda x, a, b, c, d, e: monoexp_line_step(x,a,b,c,d,e,t_sig_onset), time, ROIs_means[:,j], bounds=(0, upper_bounds))
                    res_explinestep = ROIs_means[:,j] - monoexp_line_step(time, *np.insert(popt3,len(popt3),t_sig_onset))
                    total_res_explinestep = np.sum(abs(res_explinestep**2))/len(ROIs_means[:,j])
                    if np.all(np.greater(upper_bounds,popt3+0.01*(popt3)))==True:
                        clip_flag = 0
                    #Breaks if either exponential coefficient or linear coefficients "overflow"
                    if (popt3[0]+popt3[3]>1.2*y0)|(popt3[1]<0.05):
                        break

        #subtracts fit
        if ((transitory==True) | (transitory==None)):
            corrected[:,j] = ROIs_means[:,j] - monoexp_and_line(time, *popt2)
        else:
            #exclude step in order to leave just photobleaching
            steps[j] = np.copy(-2*popt3[-1])
            popt3[-1] = 0
            corrected[:,j] = ROIs_means[:,j] - monoexp_line_step(time, *np.insert(popt3,len(popt3),t_sig_onset))
        #First frame gets median to avoid some cases where exponential function is assigned almost exclusevely to the first frame
        corrected[0,:] = np.mean(corrected[idx_msk,:],axis=0)

        corrections=None #obsolete
    return(corrected, corrections)

'''#########################################################################
                        Remove Bleaching Main Function
   #########################################################################'''

def photob_remove(ROIs_means, time, video: ImageData, label_image: LabelsData, transitory: bool=True,t_sig_onset: float=None,t_sig_end: float=None):
    #Create vector masks indicating time periods of inactivity
    if t_sig_onset==t_sig_end:
        t_sig_onset = None
        t_sig_end = None
    mask = label_image.data
    inactive_msk,t_sig_onset,t_sig_end = create_inactive_idx_msk(video,time,mask,transitory,t_sig_onset=t_sig_onset,t_sig_end=t_sig_end)

    ROIs_means_corrected, corrections = photob_fit(ROIs_means,time,inactive_msk,transitory,t_sig_onset)

    return(ROIs_means_corrected, inactive_msk, t_sig_onset, t_sig_end)
