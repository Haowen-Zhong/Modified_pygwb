import numpy as np
import scipy.ndimage as ndi
from loguru import logger

from pygwb.constants import H0
from pygwb.omega_spectra import OmegaSpectrum

from .util import _check_omegaspectra, calc_bias, window_factors, coarse_grain_window_factors

# old name: postprocess_Y_sigma; New name: postprocess_Yf_sigmaf
# NEW FUNCTION TO GET FINAL NARROWBAND PT EST AND VAR
# This function should ONLY be used to combine narrowband estimators in a single job to get a final optimal narrowband estimator for a job
def postprocess_Yf_sigmaf(
    Y_fs,
    var_fs,
    segment_duration,
    deltaF,
    new_sample_rate,
    frequency_mask=True,
    badtimes_mask=None,
    window_fftgram_dict={"window_fftgram": "hann"},
    window_fftgram_dict_welch={"window_fftgram": "hann"},
    overlap_factor=0.5,
    overlap_factor_welch=0.5,
    N_avg_segs=2,
):
    """Run postprocessing of point estimate and sigma spectrograms, combining even and
    odd segments in the case of overlapping data. For more details see - arxiv 2606.23816

    Parameters:
    -----------
    Y_fs : array-like
        2D array of point estimates with Ntimes x Nfreqs with overlapping segments
    var_fs : array-like
        2D array of variances or 2D with dimensions Ntimes x Nfreqs with overlapping time segments
    segment_duration : float
        Duration of each time segment
    deltaF : float
        Frequency resolution
    new_sample_rate : float
        sample rate of timeseries after resampling
    frequency_mask: array-like, optional
        Boolean mask to apply to frequencies for the calculation. Defaults to True which includes all frequencies in the analysis.
    badtimes_mask: array-like, optional
        Boolean mask to apply to GPStimes in the calculation. Defaults to None such that all times are included.
    window_fftgram_dict: dictionary, optional
        Dictionary with window characteristics used in PSD estimation. Default is `(window_fftgram_dict={"window_fftgram": "hann"}`
    overlap_factor: float, optional
        Overlap factor used in PSD estimation. Default is 0.5.

    Returns:
    --------
    Y_f_new : array-like
        1D point estimate spectrum
    sigma_f_few : array-like
        1D sigma spectrum
    """
    if badtimes_mask is None:
        badtimes_mask = np.zeros(len(Y_fs), dtype=bool)

    goodtimes_mask = ~badtimes_mask
    labels, n_labels = ndi.label(goodtimes_mask)

    Y_fs_sliced = []
    var_fs_sliced = []
    

    for sli in ndi.find_objects(labels):
        bias = calc_bias(
        segment_duration,
        deltaF,
        1 / new_sample_rate,
        N_avg_segs=N_avg_segs,
        window_fftgram_dict=window_fftgram_dict_welch,
        overlap_factor=overlap_factor_welch,
         )
        logger.debug(f"Bias factor: {bias}")
        

        Y = Y_fs[sli]
        var = var_fs[sli]

        if len(Y) == 1:
            Y_fs_sliced.append(Y[0])
            var_fs_sliced.append(var[0])
        else:
            # old: odd_even_segment_postprocessing, new: odd_even_segment_postprocessing_f
            # Be careful here! One has to use different ways to combine spectrograms to get optimal narrowband and broadband estimators!
            Y_red, var_red = odd_even_segment_postprocessing_f(
                Y,
                var, 
                segment_duration,
                deltaF, 
                new_sample_rate,
                frequency_mask=frequency_mask,
                window_fftgram_dict=window_fftgram_dict,
                overlap_factor=overlap_factor,
                N_avg_segs=N_avg_segs,
            )
            Y_fs_sliced.append(Y_red)
            var_fs_sliced.append(var_red)

    Y_fs_sliced = np.array(Y_fs_sliced)
    var_fs_sliced = np.array(var_fs_sliced)
    # different slices of data are independent! One can combine them by the inverse-variance weighted average as usual.
    Y_f_new, sigma_f_new = combine_spectra_with_sigma_weights(
        Y_fs_sliced, np.sqrt(var_fs_sliced)
    )

    return Y_f_new, sigma_f_new
    
# NEW FUNCTION TO GET FINAL BROADBAND PT EST AND VAR
# This function should ONLY be used to compute the optimal broadband estimator and variance of a single job!   
def postprocess_Y_sigma(
    Y_fs,
    var_fs,
    segment_duration,
    deltaF,
    new_sample_rate,
    frequency_mask=True,
    badtimes_mask=None,
    window_fftgram_dict={"window_fftgram": "hann"},
    window_fftgram_dict_welch={"window_fftgram": "hann"},
    overlap_factor=0.5,
    overlap_factor_welch=0.5,
    N_avg_segs=2,
):
    """Run postprocessing of point estimate and sigma spectrograms, combining even and
    odd segments in the case of overlapping data.
    For more details see - https://dcc.ligo.org/public/0027/T040089/000/T040089-00.pdf

    Parameters
    =======
    Y_fs: ``array-like``
        2D array of point estimates with Ntimes x Nfreqs with overlapping segments.
    var_fs: ``array-like``
        2D array of variances or 2D with dimensions Ntimes x Nfreqs with overlapping time segments.
    segment_duration: ``float``
        Duration of each time segment.
    deltaF: ``float``
        Frequency resolution.
    new_sample_rate: ``float``
        Sample rate of timeseries after resampling.
    frequency_mask: ``array-like``, optional
        Boolean mask to apply to frequencies for the calculation.
        Defaults to True which includes all frequencies in the analysis.
    badtimes_mask: ``array-like``, optional
        Boolean mask to apply to GPStimes in the calculation. Defaults to None such that all times are included.
    window_fftgram_dict: ``dictionary``, optional
        Dictionary with window characteristics used in PSD estimation.
        Default is ``window_fftgram_dict={"window_fftgram": "hann"}``
    overlap_factor: ``float``, optional
        Overlap factor used in PSD estimation. Default is 0.5.
    N_avg_segs: ``int``, optional
        Number of segments over which the average is performed.
        This is useful for computing the bias, nothing more. Default is 2.

    Returns
    =======
    Y_f_new: ``array-like``
        1D point estimate spectrum.
    sigma_f_few: ``array-like``
        1D sigma spectrum.

    See also
    --------
    pygwb.util.calc_bias
    """
    if badtimes_mask is None:
        badtimes_mask = np.zeros(len(Y_fs), dtype=bool)

    goodtimes_mask = ~badtimes_mask
    labels, n_labels = ndi.label(goodtimes_mask)

    Y_sliced = []
    var_sliced = []

    for sli in ndi.find_objects(labels):
        bias = calc_bias(
        segment_duration,
        deltaF,
        1 / new_sample_rate,
        N_avg_segs=N_avg_segs,
        window_fftgram_dict=window_fftgram_dict_welch,
        overlap_factor=overlap_factor_welch,
         )
        logger.debug(f"Bias factor: {bias}")


        Y = Y_fs[sli]
        var = var_fs[sli]

        if len(Y) == 1:
            Y_sliced.append(Y[0])
            var_sliced.append(var[0])
        else:
            Y_red, var_red = odd_even_segment_postprocessing(
                Y,
                var,
                segment_duration,
                deltaF, # newly added
                new_sample_rate,
                frequency_mask=frequency_mask,
                window_fftgram_dict=window_fftgram_dict,
                overlap_factor=overlap_factor,
                N_avg_segs=N_avg_segs,
            )
            Y_sliced.append(Y_red)
            var_sliced.append(var_red)

    Y_sliced = np.array(Y_sliced)
    var_sliced = np.array(var_sliced)

    Y_new = np.sum(Y_sliced/var_sliced)/np.sum(1/var_sliced)
    sigma_new = 1/np.sqrt(np.sum(1/var_sliced))

    # If one uses data to esitmate PSDs, then one needs to multiply sigma_new by the bias factor
    # if one uses the *true* PSDs, then no need to multiply it by the bias factor
    sigma_new = sigma_new # * bias

    return Y_new, sigma_new

# Used to combine narrowband estimators
def odd_even_segment_postprocessing_f(
    Y_fs,
    var_fs,
    segment_duration,
    deltaF, # newly added
    new_sample_rate,
    frequency_mask=True,
    window_fftgram_dict={"window_fftgram": "hann"},
    overlap_factor=0.5,
    N_avg_segs=2,
):
    """Perform averaging which combines even and odd segments for overlapping data. 

    Parameters:
    -----------
    Y_fs : array-like
        2D array of point estimates with Ntimes x Nfreqs with overlapping segments
    var_fs : array-like
        2D array of variances or 2D with dimensions Ntimes x Nfreqs with overlapping time segments
    segment_duration : float
        Duration of each time segment
    new_sample_rate : float
        sample rate of timeseries after resampling
    frequency_mask: array-like, optional
        Boolean mask to apply to frequencies for the calculation
    window_fftgram_dict: dictionary, optional
        Dictionary with window characteristics used in PSD estimation. Default is `(window_fftgram_dict={"window_fftgram": "hann"}`
    overlap_factor: float, optional

    Returns:
    --------
    Y_f_new : array-like
        1D point estimate spectrum
    var_f_few : array-like
        1D sigma spectrum
    """
    w1w2bar, _, w1w2ovlbar, _ = window_factors(
        int(segment_duration * new_sample_rate), window_fftgram_dict, overlap_factor=overlap_factor
    )
    
    # zeropad M->2M
    M = 2*int(segment_duration * deltaF)
    if M==1:
        w1w2curlybarsquared, w1w2curlyovlbarsquared = w1w2bar**2, w1w2ovlbar**2
    else:
        w1w2curlybarsquared, w1w2curlyovlbarsquared = coarse_grain_window_factors(M, 
        16384, window_fftgram_dict, overlap_factor=overlap_factor
        )
    O = overlap_factor
    k_nb = w1w2curlyovlbarsquared/w1w2curlybarsquared
    size = np.size(Y_fs, axis=0)
    # even/odd indices
    evens = np.arange(0, size, 2)
    odds = np.arange(1, size, 2)

    C_even = np.nansum(Y_fs[evens] / var_fs[evens], axis=0)/np.nansum(var_fs[evens] ** -1, axis=0)
    C_odd = np.nansum(Y_fs[odds] / var_fs[odds], axis=0)/np.nansum(var_fs[odds] ** -1, axis=0)
    sigma2_even = 1/np.nansum(var_fs[evens] ** -1, axis=0)
    sigma2_odd = 1/np.nansum(var_fs[odds] ** -1, axis=0)
    # X_even = np.nansum(Y_fs[evens] / var_fs[evens], axis=0)
    # GAMMA_even = np.nansum(var_fs[evens] ** -1, axis=0)
    # X_odd = np.nansum(Y_fs[odds] / var_fs[odds], axis=0)
    # GAMMA_odd = np.nansum(var_fs[odds] ** -1, axis=0)
    sigma2_1 = var_fs[0, :]
    sigma2_N = var_fs[-1, :]
    sigma2bar = 1/sigma2_even + 1/sigma2_odd - (1/2)* (1 / sigma2_1 + 1 / sigma2_N)
    #sigma2_odd, sigma2_even, sigma2_1, sigma2_N = [ s if s!=np.inf else 0 for s in (sigma2_odd, sigma2_even, sigma2_1, sigma2_N)]
    #sigma2_1[sigma2_1==np.inf]=0
    #sigma2_N[sigma2_N==np.inf]=0
    #sigma2_even[sigma2_even==np.inf]=0
    #sigma2_odd[sigma2_odd==np.inf]=0


    Y_f_new = (
        C_odd/sigma2_odd * (1 - O**2*k_nb * sigma2_odd * sigma2bar) + C_even/sigma2_even * (1 - O**2*k_nb * sigma2_even * sigma2bar))/ (
        1/sigma2_odd
        + 1/sigma2_even
        - 2*O**2*k_nb*sigma2bar
    )

    var_f_new = (1 - O**4 * k_nb**2 * sigma2_odd * sigma2_even * sigma2bar ** 2)/ (
        1/sigma2_odd
        + 1/sigma2_even
        - 2*O**2*k_nb*sigma2bar
    ) 

    return Y_f_new, var_f_new

# Different funtion! To combine broadband pst and var
def odd_even_segment_postprocessing(
    Y_fs,
    var_fs,
    segment_duration,
    deltaF, #deltaF is needed to calculate sigma_oo, sigma_ee
    new_sample_rate,
    frequency_mask=True,
    window_fftgram_dict={"window_fftgram": "hann"},
    overlap_factor=0.5,
    N_avg_segs=2,
):
    """Perform averaging which combines even and odd segments for overlapping data. 

    Parameters
    =======
    Y_fs: ``array-like``
        2D array of point estimates with Ntimes x Nfreqs with overlapping segments.
    var_fs: ``array-like``
        2D array of variances or 2D with dimensions Ntimes x Nfreqs with overlapping time segments.
    segment_duration: ``float``
        Duration of each time segment.
    new_sample_rate: ``float``
        Sample rate of timeseries after resampling.
    frequency_mask: ``array-like``, optional
        Boolean mask to apply to frequencies for the calculation.
    window_fftgram_dict: ``dictionary``, optional
        Dictionary with window characteristics used in PSD estimation.
        Default is ``window_fftgram_dict={"window_fftgram": "hann"}``.
    overlap_factor: ``float``, optional
        Defines the overlap between consecutive data chunks used in the calculation. Default is 0.5.
    
    Returns
    =======
    Y_f_new: ``array-like``
        1D point estimate spectrum.
    var_f_few: ``array-like``
        1D sigma spectrum.

    See also
    --------
    pygwb.util.window_factors
    """
    _, w1w2squaredbar, _, w1w2squaredovlbar = window_factors(
        int(segment_duration * new_sample_rate), window_fftgram_dict, overlap_factor=overlap_factor
    )
    # same as above
    M = 2*int(segment_duration * deltaF)
    w1w2curlybarsquared, _ = coarse_grain_window_factors(M, 
        int(segment_duration * new_sample_rate), window_fftgram_dict, overlap_factor=overlap_factor
    )
    W_curly = w1w2squaredbar/w1w2curlybarsquared 
    O = overlap_factor
    k_bb = w1w2squaredovlbar / w1w2squaredbar # k_bb is the same as the k factor in original code
    size = np.size(Y_fs, axis=0)
    # even/odd indices
    evens = np.arange(0, size, 2)
    odds = np.arange(1, size, 2)



    X_even = np.nansum(Y_fs[evens] / var_fs[evens], axis=0)
    GAMMA_even = np.nansum(var_fs[evens] ** -1, axis=0)
    X_odd = np.nansum(Y_fs[odds] / var_fs[odds], axis=0)
    GAMMA_odd = np.nansum(var_fs[odds] ** -1, axis=0)

    frequency_mask = np.ones_like(frequency_mask) 
    C_odd  = np.nansum(X_odd[frequency_mask])/np.nansum(GAMMA_odd[frequency_mask])
    C_even = np.nansum(X_even[frequency_mask])/np.nansum(GAMMA_even[frequency_mask])

    sigma2_oo = 2*W_curly/np.nansum(GAMMA_odd[frequency_mask])/M
    sigma2_ee = 2*W_curly/np.nansum(GAMMA_even[frequency_mask])/M
    sigma2_1 = 2*W_curly/np.nansum(var_fs[0, frequency_mask] ** -1)/M
    sigma2_N = 2*W_curly/np.nansum(var_fs[-1, frequency_mask] ** -1)/M
    sigma2bar = 1 / sigma2_oo + 1 / sigma2_ee - (1 / 2) * (1 / sigma2_1 + 1 / sigma2_N)
    
    sigma2_oo, sigma2_ee, sigma2_1, sigma2_N = [
        s if s != np.inf else 0 for s in (sigma2_oo, sigma2_ee, sigma2_1, sigma2_N)
    ]

    Y_new = (
        C_odd * (sigma2_oo**(-1) - O*k_bb * sigma2bar)
        + C_even * (sigma2_ee**(-1) - O*k_bb * sigma2bar)
    ) / (sigma2_oo**(-1)+ sigma2_ee**(-1) - 2*O*k_bb* sigma2bar
    )

    inv_var_new = (
        sigma2_oo**(-1) + sigma2_ee**(-1) - 2*O*k_bb*sigma2bar
        ) / (1 - O**2*k_bb**2 * sigma2_oo * sigma2_ee * sigma2bar ** 2)
    

    var_new = 1 / inv_var_new

    return Y_new, var_new


# This is the old function that will NOT be used!
def calc_Y_sigma_from_Yf_sigmaf(
    Y_f, sigma_f, frequency_mask=True, alpha=None, fref=None
):
    """
    Calculate the omega point estimate and sigma from their respective spectra,
    or spectrograms, taking into account the desired spectral weighting.
    To apply weighting, the frequency array associated to the spectra must be supplied.

    If applied to a 1D array, you get single numbers out. If applied to a 2D array, it combines
    over the second dimension. That is, if dimension is Ntimes x Nfrequencies, then the resulting
    spectra are Ntimes long.

    Parameters
    ==========
    Y_f: `pygwb.omega_spectrogram.OmegaSpectrogram`
        Point estimate spectrum
    sigma_f: `pygwb.omega_spectrogram.OmegaSpectrogram`
        Sigma spectrum
    frequency_mask: array-like, optional
        Boolean mask to apply to frequencies for the calculation.
    alpha: float, optional
        Spectral index to use in case re-weighting is requested.
    fref: float, optional
        Reference frequency to use in case re-weighting is requested.

    Returns:
    --------
    Y : array-like or float
        Point estimate or Point estimate spectrum
    sigma : array-like or float
        point estimate standard deviation (theoretical) or spectrum of point estimate
        standard deviations
    Note
    ====
    If passing in spectrograms, the point estimate and sigma will be calculated per
    spectrum, without any time-averaging applied.
    Y_f and sigma_f can also be gwpy.Spectrogram objects, or numpy arrays. In these cases
    however the reweight functionality is not supported.

    """
    if alpha is not None or fref is not None:
        Y_f.reweight(new_alpha=alpha, new_fref=fref)
        sigma_f.reweight(new_alpha=alpha, new_fref=fref)

    # now just strip off what we need...
    try:
        Y_f = np.real(Y_f.value)
        var_f = sigma_f.value ** 2
    except AttributeError:
        Y_f = np.real(Y_f)
        var_f = sigma_f ** 2

    if isinstance(frequency_mask, np.ndarray):
        pass
    elif frequency_mask == True:
        if len(Y_f.shape) == 1:
            frequency_mask = np.ones(Y_f.shape[0], dtype=bool)
        elif len(Y_f.shape) == 2:
            frequency_mask = np.ones(Y_f.shape[1], dtype=bool)
    if len(Y_f.shape) == 1 or Y_f.shape[0] == 1:
        if Y_f.shape[0] == 1:
            Y_f = Y_f[0]
            var_f = var_f[0]
        var = 1 / np.sum(var_f[:] ** (-1), axis=-1).squeeze()
        Y = np.nansum(Y_f[:] * (var / var_f[:]), axis=-1)
    # need to make this nan-safe
    elif len(Y_f.shape) == 2:
        var = 1 / np.sum(var_f[:, :] ** (-1), axis=-1).squeeze()
        Y = np.einsum(
                "tf, t -> t", Y_f[:, :] / var_f[:, :], var
        )
    else:
        raise ValueError("The input is neither a spectrum nor a spectrogram.")

    sigma = np.sqrt(var)

    return Y, sigma

# This function is used to compute broadband estimator and var for a single SEGMENT!
def calc_Y_sigma_from_Yf_sigmaf_s(
    Y_f, sigma_f, 
    segment_duration,
    deltaF, # newly added
    new_sample_rate,
    frequency_mask=True,
    window_fftgram_dict={"window_fftgram": "hann"},
    overlap_factor=0.5,
    N_avg_segs=2,
    alpha=None, 
    fref=None
):
    """
    Calculate the omega point estimate and sigma from their respective spectra,
    or spectrograms, taking into account the desired spectral weighting.
    To apply weighting, the frequency array associated to the spectra must be supplied.

    If applied to a 1D array, you get single numbers out. If applied to a 2D array, it combines
    over the second dimension. That is, if dimension is Ntimes x Nfrequencies, then the resulting
    spectra are Ntimes long.

    Parameters
    =======
    Y_f: ``pygwb.omega_spectrogram.OmegaSpectrogram``
        Point estimate spectrum.
    sigma_f: ``pygwb.omega_spectrogram.OmegaSpectrogram``
        Sigma spectrum.
    frequency_mask: ``array-like``, optional
        Boolean mask to apply to frequencies for the calculation. Default set to True including all frequencies.
    alpha: ``float``, optional
        Spectral index to use in case re-weighting is requested. Default set to None.
    fref: ``float``, optional
        Reference frequency to use in case re-weighting is requested. Default set to None.

    Returns
    =======
    Y: ``array-like`` or ``float``
        Point estimate or Point estimate spectrum.
    sigma: ``array-like`` or ``float``
        Point estimate standard deviation (theoretical) or spectrum of point estimate standard deviations.

    Notes
    -----
    If passing in spectrograms, the point estimate and sigma will be calculated per
    spectrum, without any time-averaging applied.
    Y_f and sigma_f can also be ``gwpy.spectrogram.Spectrogram`` objects, or numpy arrays. In these cases
    however the reweight functionality is not supported.

    """
    _, w4bar, _, _ = window_factors(
       int(segment_duration * new_sample_rate), window_fftgram_dict, overlap_factor=overlap_factor
    )
    w1w2curlybarsquared, _ = coarse_grain_window_factors(2*int(segment_duration * deltaF), 
        int(segment_duration * new_sample_rate), window_fftgram_dict, overlap_factor=overlap_factor
    )
    W_curly = w4bar/w1w2curlybarsquared
    M = 2*segment_duration * deltaF
    # Reweight in case one wants to pass it.
    if alpha is not None or fref is not None:
        Y_f.reweight(new_alpha=alpha, new_fref=fref)
        sigma_f.reweight(new_alpha=alpha, new_fref=fref)

    # now just strip off what we need...
    try:
        Y_f = np.real(Y_f.value)
        var_f = sigma_f.value ** 2
    except AttributeError:
        Y_f = np.real(Y_f)
        var_f = sigma_f ** 2

    if isinstance(frequency_mask, np.ndarray):
        pass
    elif frequency_mask == True:
        if len(Y_f.shape) == 1:
            frequency_mask = np.ones(Y_f.shape[0], dtype=bool)
        elif len(Y_f.shape) == 2:
            frequency_mask = np.ones(Y_f.shape[1], dtype=bool)
    if len(Y_f.shape) == 1 or Y_f.shape[0] == 1:
        if Y_f.shape[0] == 1:
            Y_f = Y_f[0]
            var_f = var_f[0]
        var = 1 / np.sum(var_f[frequency_mask] ** (-1), axis=-1).squeeze()
        Y = np.nansum(Y_f[frequency_mask] * (var / var_f[frequency_mask]), axis=-1)
    # need to make this nan-safe
    elif len(Y_f.shape) == 2:
        frequency_mask = np.ones_like(frequency_mask)
        var = 1 / np.sum(var_f[:, frequency_mask] ** (-1), axis=-1).squeeze()
        Y = np.einsum(
            "tf, t -> t", Y_f[:, frequency_mask] / var_f[:, frequency_mask], var
        )
    else:
        raise ValueError("The input is neither a spectrum nor a spectrogram.")
    sigma = np.sqrt(2*W_curly*var/M)

    return Y, sigma


def calculate_point_estimate_sigma_spectra(
    freqs,
    csd,
    avg_psd_1,
    avg_psd_2,
    orf,
    sample_rate,
    segment_duration,
    window_fftgram_dict={"window_fftgram": "hann"},
    overlap_factor=0.5,
    fref=25.0,
    alpha=0.0,
):
    """
    Calculate the Omega point estimate and associated sigma integrand,
    given a set of cross-spectral and power-spectral density spectrograms.
    This is particularly useful for statistical checks.

    If CSD is set to None, only returns variance.

    Parameters
    ==========
    freqs: array_like
        Frequencies associated to the spectrograms.
    csd: gwpy Spectrogram
        CSD spectrogram for detectors 1 and 2.
    avg_psd_1: gwpy Spectrogram
        Spectrogram of averaged PSDs for detector 1.
    avg_psd_2: gwpy Spectrogram
        Spectrogram of averaged PSDs for detector 2.
    orf: array_like
        Overlap reduction function.
    sample_rate: float
        Sampling rate of the data.
    segment_duration: float
        Duration of each segment in seconds.
    window_fftgram_dict: dictionary, optional
        Dictionary with window characteristics used in analysis segment estimation. Default is `(window_fftgram_dict={"window_fftgram": "hann"}`
    overlap_factor: float, optional
        Overlap factor used in analysis segment estimation. Default is 0.5.
    fref: float, optional
        Reference frequency to use in the weighting calculation.
        Final result refers to this frequency.
    alpha: float, optional
        Spectral index to use in the weighting.
    """
    S_alpha = 3 * H0.si.value ** 2 / (10 * np.pi ** 2) / freqs ** 3
    S_alpha *= (freqs / fref) ** float(alpha)

    var_fs = avg_psd_1 * avg_psd_2 / (2 * orf ** 2 * S_alpha ** 2) # One CANNOT just divide it by M!
    
    w1w2bar, _, _, _ = window_factors(
        int(sample_rate * segment_duration), window_fftgram_dict=window_fftgram_dict, overlap_factor=overlap_factor
    )
    # zeropad: M->2M
    M = 2*int(segment_duration * (freqs[1] - freqs[0]))

    w1w2curlybarsquared, _ = coarse_grain_window_factors(M, 
        16384, window_fftgram_dict, overlap_factor=overlap_factor
        )

    r_of_M = w1w2curlybarsquared/w1w2bar**2 
    var_fs = var_fs * r_of_M
    if csd is not None:
        Y_fs = (csd) / (orf * S_alpha)
        return Y_fs, var_fs
    else:
        return var_fs


def combine_spectra_with_sigma_weights(main_spectra, weights_spectra):
    r"""
    Combine different statistically independent spectra :math:`S_i(f)` using spectral weights :math:`w_i(f)`, as


    .. math::

        S(f) = \frac{\sum_i \frac{S_i(f)}{w^2_i(f)}}{\sum_i \frac{1}{w^2_i(f)}},\,\,\,\, \sigma = \sqrt{\frac{1}{\sum_i \frac{1}{w^2_i(f)}}}.


    If main_spectra is 2D and has dimensions N_1 x N_2, final spectrum has dimension N_2 (in contrast to `calc_Y_sigma_from_Yf_sigmaf`
    which combines across other dimension).

    Parameters
    =========
    main_spectra: np.array
        Array of arrays or FrequencySeries or OmegaSpectrum objects to be combined.
    weights_spectra: np.array
        Array of arrays or FrequencySeries or OmegaSpectrum objects to use as weights.

    Returns
    =======
    combined_weighted_spectrum: array_like
        Final spectrum obtained combining the original spectra with given weights.
    combined_weights_spectrum: array_like
        Variance associated to the final spectrum obtained combining the given weights.
    """
    if isinstance(main_spectra[0], OmegaSpectrum):
        _check_omegaspectra(main_spectra)

    if isinstance(weights_spectra[0], OmegaSpectrum):
        _check_omegaspectra(weights_spectra)

    w_spec = np.array(weights_spectra)
    m_spec = np.array(main_spectra)
    res_1 = 1 / np.nansum(1 / w_spec ** 2, axis=0)
    combined_weights_spectrum = np.sqrt(res_1)
    combined_weighted_spectrum = (
        np.nansum(m_spec / w_spec ** 2, axis=0) * res_1
    )
    if isinstance(main_spectra[0], OmegaSpectrum):
        combined_weighted_omegaspectrum = OmegaSpectrum(combined_weighted_spectrum, alpha=main_spectra[0].alpha, fref=main_spectra[0].fref, h0=main_spectra[0].h0, frequencies=main_spectra[0].frequencies, name='omega_spectrum')
        combined_weights_omegaspectrum = OmegaSpectrum(combined_weights_spectrum, alpha=main_spectra[0].alpha, fref=main_spectra[0].fref, h0=main_spectra[0].h0, frequencies=main_spectra[0].frequencies, name='sigma_spectrum')
        return combined_weighted_omegaspectrum, combined_weights_omegaspectrum
    else:
         return combined_weighted_spectrum, combined_weights_spectrum
