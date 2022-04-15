
import scipy.signal
import numpy as np

def gen_lpf(cutoff, fs, numtaps=512):
    fny = fs/2.0
    coefs = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff/fny)
    return coefs


def shift_lpf(coefs, fnew, fs):
    mod = 2.0 * np.cos(2.0 * np.pi * fnew / fs * np.arange(len(coefs)))
    return coefs * mod


def generate_filters(bw, num, fs, numtaps=512):
    mother_coefs = gen_lpf(bw/2.0, fs, numtaps=numtaps)

    filter_banks = [
        shift_lpf(mother_coefs, bw + k*bw, fs)
        for k in range(num-1)
    ]

    filter_freqs = [
        bw * k
        for k in range(num)
    ]
    filter_banks.insert(0, mother_coefs)
    return (filter_banks, filter_freqs)
