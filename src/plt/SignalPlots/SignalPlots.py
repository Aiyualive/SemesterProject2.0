t_n = 1
N = 24000
T = t_n / N
f_s = 1/T

from scipy.signal import welch
from scipy.fftpack import fft
import pywt
import matplotlib.pyplot as plt
import numpy as np
"""
    Derived from http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
"""

### Add EMD
### add title
### add metrics below plot
def _get_ave_values(xvalues, yvalues, n = 5):
    """
    """
    signal_length = len(xvalues)
    if signal_length % n == 0:
        padding_length = 0
    else:
        padding_length = n - signal_length//n % n
    xarr = np.array(xvalues)
    yarr = np.array(yvalues)
    xarr.resize(signal_length//n, n)
    yarr.resize(signal_length//n, n)
    xarr_reshaped = xarr.reshape((-1,n))
    yarr_reshaped = yarr.reshape((-1,n))
    x_ave = xarr_reshaped[:,0]
    y_ave = np.nanmean(yarr_reshaped, axis=1)
    return x_ave, y_ave

def _get_fft_values(y_values, T, N, f_s):
    """
    Fast Fourier Transform
    """
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def _get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

def _autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]

def _get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = _autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values

def _lowpassfilter(signal, thresh = 0.63, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

def _waveletfilter(sig, wavelet='db8', decomposition_level=8, filt_upper_level=6, filt_lower_level=1):
    if filt_upper_level<filt_lower_level:
        raise Warning('wong filter assignemnt levels reconstruction')
    # Decomposition
    coeffs = pywt.wavedec(sig,wavelet,level=decomposition_level)
    # Filtering out the low frequency stuff by setting cDi where i>reclvl
    for i in range(1,1+decomposition_level-filt_upper_level):
        coeffs[i]=coeffs[i]*0
    # Removing lower levels (if 1 keeps all including the first one)
    for i in range(1,filt_lower_level):
        coeffs.pop()#
    # Reconstructing signal
    sig_reconstructed = pywt.waverec(coeffs,wavelet)
    return sig_reconstructed

def plot_window(df, savefigure = False, save_path=""):
    for name, v in df.iterrows():
        fig = plt.figure(figsize=(20,5), constrained_layout=True)

        x, y = (v['timestamps'], v['accelerations'])
        plt.plot(x, y, label='signal')

        #x_avg, y_avg = get_ave_values(x, y)
        #plt.plot(x_avg, y_avg, label = 'time average (n={})'.format(5))

        plt.title(name, fontsize=16)
        plt.xlabel('time', fontsize=16)
        plt.ylabel('Acceleratioin', fontsize=16)
        plt.legend()

        fig.set_size_inches(20, 5, forward=True)
        plt.show()
        if savefigure == True:
            fig.savefig('AiyuDocs/test_imgs/raw_' + name,  bbox_inches='tight')


def plot_fft(df, savefigure = False, save_path=""):
    for name, v in df.iterrows():
        fig = plt.figure(figsize=(20,5), constrained_layout=True)

        acc = v['accelerations']
        f_values, fft_values = _get_fft_values(acc, T, N, f_s)
        plt.plot(f_values, fft_values, 'r-', label='Fourier Transform')

        # variance = np.std(acc)**2
        # fft_power = variance * abs(fft_values) ** 2
        # plt.plot(f_values, fft_power, 'k--', linewidth=1, label='FFT Power Spectrum')

        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.title(name, fontsize=16)
        plt.legend()

        fig.set_size_inches(20, 5, forward=True)
        plt.show()
        if savefigure == True:
            fig.savefig('AiyuDocs/test_imgs/ft-power_' + name,  bbox_inches='tight')

def plot_psd(df, savefigure = False, save_path=""):
    for name, v in df.iterrows():
        fig = plt.figure(figsize=(20,5), constrained_layout=True)

        acc = v['accelerations']
        f_values, psd_values = _get_psd_values(acc, T, N, f_s)
        plt.plot(f_values, psd_values, linestyle='-', color='blue')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2 / Hz]')
        plt.title(name, fontsize=16)

        fig.set_size_inches(20, 5, forward=True)
        plt.show()
        if savefigure == True:
            fig.savefig('AiyuDocs/test_imgs/psd_' + name,  bbox_inches='tight')

def plot_autocorr(df, savefigure = False, save_path=""):
    for name, v in df.iterrows():
        fig = plt.figure(figsize=(20,5), constrained_layout=True)

        acc = v['accelerations']
        t_values, autocorr_values = _get_autocorr_values(acc, T, N, f_s)
        plt.plot(t_values, autocorr_values, linestyle='-', color='blue')
        plt.xlabel('time delay [s]')
        plt.ylabel('Autocorrelation amplitude')
        plt.title(name, fontsize=16)

        fig.set_size_inches(20, 5, forward=True)
        plt.show()
        if savefigure == True:
            fig.savefig('AiyuDocs/test_imgs/corr_' + name,  bbox_inches='tight')

def plot_wavelet(df, savefigure=False, save_path="", waveletname = 'cmor1-1.5', cmap = plt.cm.seismic):

    scales = np.arange(1, 128)
    t0=0
    dt=T
    time = np.arange(0, N) * dt + t0

    for name, v in df.iterrows():
        fig, ax = plt.subplots(figsize=(20, 5))
        #plot_wavelet(ax, time, v, scales, xlabel=xlabel, ylabel=ylabel, title=name)

        acc = v['accelerations']
        dt = time[1] - time[0]
        [coefficients, frequencies] = pywt.cwt(acc, scales, waveletname, dt)
        power = (abs(coefficients)) ** 2
        period = 1. / frequencies
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
        contourlevels = np.log2(levels)
        im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
        ax.set_title('Wavelet Transform (Power Spectrum) of signal', fontsize=20)
        ax.set_ylabel('Period', fontsize=18)
        ax.set_xlabel('Time', fontsize=18)

        yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels(yticks)
        ax.invert_yaxis()
        ylim = ax.get_ylim()

        fig.set_size_inches(20, 5, forward=True)
        plt.show()
        if savefigure == True:
            fig.savefig('AiyuDocs/test_imgs/wavelet_' + name,  bbox_inches='tight')

def plot_lowpass(df, savefigure=False, save_path=""):
    new = pd.DataFrame()
    for name, v in df.iterrows():
        acc = v['accelerations']
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(acc, color="b", alpha=0.5, label='original signal')

        rec = _lowpassfilter(acc, 0.4)
        temp_df = pd.DataFrame([rec],
                                index   = [name],
                                columns = ['accelerations'])
        new = pd.concat([new, temp_df], axis=0)

        ax.plot(rec, 'k', label='DWT smoothing}', linewidth=2)
        ax.legend()
        ax.set_title('Removing High Frequency Noise with DWT', fontsize=18)
        ax.set_ylabel('Signal Amplitude', fontsize=16)
        ax.set_xlabel('Sample No', fontsize=16)
        fig.set_size_inches(20, 5, forward=True)
        plt.show()

        if savefigure == True:
            fig.savefig('AiyuDocs/test_imgs/lowpass_' + name,  bbox_inches='tight')
    return new

def plot_waveletfilter(df, savefigure=False, save_path=""):
    for name, v in df.iterrows():
        acc = v['accelerations']
        filt=waveletfilter(acc)

        plt.plot(acc, label='24khz')
        plt.plot(filt,'r',label='filtered DWT')
        plt.legend()
        plt.show()

        if savefigure == True:
            fig.savefig('AiyuDocs/test_imgs/lowpass_' + name,  bbox_inches='tight')
