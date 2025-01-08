"""
Adapted from https://github.com/alsauve/scaleogram/.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from reflax.data_analysis.scaleogram.wfun import get_default_wavelet, WAVLIST, fastcwt

CBAR_DEFAULTS = {
    'vertical'   : { 'aspect':30, 'pad':0.03, 'fraction':0.05 },
    'horizontal' : { 'aspect':40, 'pad':0.12, 'fraction':0.05 }
}

COI_DEFAULTS = {
        'alpha':0.5,
        'hatch':'/',
}

CWT_FUN = fastcwt  # replacement for pywt.cwt()

class CWT:
    """Class acting as a Container for Continuous Wavelet Transform
    Allow to plot several scaleograms with the same transform

    Example::

    import scaleogram as scg
    import numpy as np

    time   = np.arange(200, dtype=np.float16)-100
    data   =  np.exp(-0.5*((time)/0.2)**2)  # insert a gaussian at the center
    scales = np.arange(1,101) # scaleogram with 100 rows

    # compute ONCE the Continuous Wavelet Transform
    cwt    = scg.CWT(time, data, scales)

    # plot 1 with full range
    scg.cws(cwt)

    # plot 2 with a zoom
    scg.cws(cwt, xlim=(-50, 50), ylim=(20, 1))

    Parameters
    ---------

    The __init__() method accept the same values and call signatures as for
    cws() function (see docstring)

    """

    def __init__(self, time, signal=None, scales=None, wavelet=None):
        # allow to build the spectrum for signal only
        if signal is None:
            signal = time
            time   = np.arange(len(time))

        # build a default scales array
        if scales is None:
            scales = np.arange(1, min(len(time)/10, 100))
        if scales[0] <= 0:
            raise ValueError("scales[0] must be > 0, found:"+str(scales[0]) )

        if wavelet is None:
            wavelet = get_default_wavelet()

        # Compute CWT
        # dt = time[1]-time[0]
        # use the mean time difference to avoid issues with irregular time steps
        dt = np.mean(np.diff(time))

        coefs, scales_freq = CWT_FUN(signal, scales, wavelet, dt)
        # Note about frequencies values:
        #   The value returned by PyWt is
        #      scales_freq = wavelet.central_frequency / scales
        #   If the time array is not provided it is expressed in
        #   Nb of oscillations over the whole signal array

        self.signal = signal
        self.time   = time
        self.scales= scales

        self.wavelet= wavelet
        self.coefs  = coefs
        self.scales_freq = scales_freq
        self.dt     = dt






def cws(time, signal=None, scales=None, wavelet=None,
         periods=None,
         spectrum='amp', coi=True, coikw=None,
         yaxis='period',
         cscale='linear', cmap='jet', clim=None,
         cbar='vertical', cbarlabel=None,
         cbarkw=None,
         xlim=None, ylim=None, yscale=None,
         xlabel=None, ylabel=None, title=None,
         figsize=None, ax=None):

    if isinstance(time, CWT):
        c = time
        time, signal, scales, dt  = c.time, c.signal, c.scales, c.dt
        coefs, scales_freq        = c.coefs, c.scales_freq
    else:
        # allow to build the spectrum for signal only
        if signal is None:
            signal = time
            time   = np.arange(len(time))

        # build a default scales array
        if scales is None:
            scales = np.arange(1, min(len(time)/10, 100))
        if scales[0] <= 0:
            raise ValueError("scales[0] must be > 0, found:"+str(scales[0]) )

        if wavelet is None:
            wavelet = get_default_wavelet()

        # wavelet transform
        dt = time[1]-time[0]
        coefs, scales_freq = CWT_FUN(signal, scales, wavelet, dt)

    # create plot area or use the one provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # adjust y axis ticks
    scales_period = 1./scales_freq  # needed also for COI mask
    xmesh = np.concatenate([time, [time[-1]+dt]])
    if yaxis == 'period':
        ymesh = np.concatenate([scales_period, [scales_period[-1]+dt]])
        ylim  = ymesh[[-1,0]] if ylim is None else ylim
        ax.set_ylabel("Period" if ylabel is None else ylabel)
    elif yaxis == 'frequency':
        df    = scales_freq[-1]/scales_freq[-2]
        ymesh = np.concatenate([scales_freq, [scales_freq[-1]*df]])
        # set a useful yscale default: the scale freqs appears evenly in logscale
        yscale = 'log' if yscale is None else yscale
        ylim   = ymesh[[-1, 0]] if ylim is None else ylim
        ax.set_ylabel("Frequency" if ylabel is None else ylabel)
        #ax.invert_yaxis()
    elif yaxis == 'scale':
        ds = scales[-1]-scales[-2]
        ymesh = np.concatenate([scales, [scales[-1] + ds]])
        ylim  = ymesh[[-1,0]] if ylim is None else ylim
        ax.set_ylabel("Scale" if ylabel is None else ylabel)
    else:
        raise ValueError("yaxis must be one of 'scale', 'frequency' or 'period', found "
                          + str(yaxis)+" instead")

    # limit of visual range
    xr = [time.min(), time.max()]
    if xlim is None:
        xlim = xr
    else:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # adjust logarithmic scales on request (set automatically in Frequency mode)
    if yscale is not None:
        ax.set_yscale(yscale)

    # choose the correct spectrum display function and name
    if spectrum == 'amp':
        values = np.abs(coefs)
        sp_title = "Amplitude"
        cbarlabel= "abs(CWT)" if cbarlabel is None else cbarlabel
    elif spectrum == 'real':
        values = np.real(coefs)
        sp_title = "Real"
        cbarlabel= "real(CWT)" if cbarlabel is None else cbarlabel
    elif spectrum == 'imag':
        values = np.imag(coefs)
        sp_title = "Imaginary"
        cbarlabel= "imaginary(CWT)" if cbarlabel is None else cbarlabel
    elif spectrum == 'power':
        sp_title = "Power"
        cbarlabel= "abs(CWT)$^2$" if cbarlabel is None else cbarlabel
        values = np.power(np.abs(coefs),2)
    elif hasattr(spectrum, '__call__'):
        sp_title = "Custom"
        values = spectrum(coefs)
    else:
        raise ValueError("The spectrum parameter must be one of 'amp', 'real', 'imag',"+
                         "'power' or a lambda() expression")

    # labels and titles
    ax.set_title("Continuous Wavelet Transform "+sp_title+" Spectrum"
                 if title is None else title)
    ax.set_xlabel("Time/spatial domain" if xlabel is None else xlabel )


    if cscale == 'log':
        isvalid = (values > 0)
        cnorm = LogNorm(values[isvalid].min(), values[isvalid].max())
    elif cscale == 'linear':
        cnorm = None
    else:
        raise ValueError("Color bar cscale should be 'linear' or 'log', got:"+
                         str(cscale))

    # plot the 2D spectrum using a pcolormesh to specify the correct Y axis
    # location at each scale
    qmesh = ax.pcolormesh(xmesh, ymesh, values, cmap=cmap, norm=cnorm)

    # plot the maximum values of the spectrum over the time axis
    # ax.plot(time, scales_period[np.argmax(values, axis=0)], 'w.')

    if clim:
        qmesh.set_clim(*clim)

    # fill visually the Cone Of Influence
    # (locations subject to invalid coefficients near the borders of data)
    if coi:
        # convert the wavelet scales frequency into time domain periodicity
        scales_coi = scales_period
        max_coi  = scales_coi[-1]

        # produce the line and the curve delimiting the COI masked area
        mid = int(len(xmesh)/2)
        time0 = np.abs(xmesh[0:mid+1]-xmesh[0])
        ymask = np.zeros(len(xmesh), dtype=np.float16)
        ymhalf= ymask[0:mid+1]  # compute the left part of the mask
        ws    = np.argsort(scales_period) # ensure np.interp() works
        minscale, maxscale = sorted(ax.get_ylim())
        if yaxis == 'period':
            ymhalf[:] = np.interp(time0,
                  scales_period[ws], scales_coi[ws])
            yborder = np.zeros(len(xmesh)) + maxscale
            ymhalf[time0 > max_coi]   = maxscale
        elif yaxis == 'frequency':
            ymhalf[:] = np.interp(time0,
                  scales_period[ws], 1./scales_coi[ws])
            yborder = np.zeros(len(xmesh)) + minscale
            ymhalf[time0 > max_coi]   = minscale
        elif yaxis == 'scale':
            ymhalf[:] = np.interp(time0, scales_coi, scales)
            yborder = np.zeros(len(xmesh)) + maxscale
            ymhalf[time0 > max_coi]   = maxscale
        else:
            raise ValueError("yaxis="+str(yaxis))

        # complete the right part of the mask by symmetry
        ymask[-mid:] = ymhalf[0:mid][::-1]

        # plot the mask and forward user parameters
        plt.plot(xmesh, ymask)
        coikw = COI_DEFAULTS if coikw is None else coikw
        ax.fill_between(xmesh, yborder, ymask, **coikw )

    # color bar stuff
    if cbar:
        cbarkw   = CBAR_DEFAULTS[cbar] if cbarkw is None else cbarkw
        colorbar = plt.colorbar(qmesh, orientation=cbar, ax=ax, **cbarkw)
        if cbarlabel:
            colorbar.set_label(cbarlabel)

    return ax