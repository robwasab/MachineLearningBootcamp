import numpy as np
import sys
import os

import matplotlib.pyplot as plt
import scipy.signal

from multiprocessing import Pool
import pdb

import argparse

def init_args():
    parser = argparse.ArgumentParser(
        usage = "Use '%s --help' for more information"%sys.argv[0]
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dir', type=str)
    group.add_argument('--file', type=str)

    args = sys.argv[1:]
    parsed_args = parser.parse_args(args)

    return vars(parsed_args)


def load_data(filepath):
    filename = os.path.split(filepath)[-1]
    label, date, time = filename.split('_')
    #print(label)
    #print(date)
    #print(time)
    return np.load(filepath), label


def calc_energy(x, fs):
    return np.sum(np.power(x, 2.0) / fs)


def normalize_energy(x, fs):
    energy = calc_energy(x, fs)
    scale = 1.0 / energy
    ret = x * np.sqrt(scale)
    return ret


def trim_signal_with_level(signal, level):
    # find start index
    for start_index in range(len(signal)-2):
        if np.abs(signal[start_index]) > level:
            break

    for stop_index in range(len(signal)-1, start_index, -1):
        if np.abs(signal[stop_index]) > level:
            break

    return signal[start_index: stop_index+1]


def trim_signal(signal, fs):
    energy = calc_energy(signal, fs)

    max_level = np.max(signal)
    min_level = 0

    while True:
        half_level = (max_level + min_level) / 2.0

        new_signal = trim_signal_with_level(signal, half_level)

        new_energy = calc_energy(new_signal, fs)

        frac = new_energy / energy

        if np.abs(max_level - min_level) < 0.001:
            break

        elif 0.99 < new_energy / energy:
            # still too much energy left over
            min_level = half_level

        elif 0.99 > new_energy / energy:
            # cut a little too much out
            max_level = half_level

        else:
            break

    return new_signal


def magnitude(x, fs, pad_to=-1):
    if pad_to > 0:
        padding = np.zeros(pad_to)
        padding[0:len(x)] = x
        x = padding
    fft = np.fft.rfft(x)
    mag = np.abs(fft)
    mag_db = np.log10(mag) * 20.0
    freqs = np.arange(len(fft)) / len(x) * fs
    return freqs, mag_db


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


def compute_energy_band(signal, bpf, fs):
    return calc_energy(
        scipy.signal.fftconvolve(signal, bpf),
        fs
    )


def measure_energy_bands(signal, filter_banks, fcenters, fs):
    columns = [
        'f%d'%fc
        for fc in fcenters
    ]
    data = [
        compute_energy_band(signal, bpf, fs)
        for bpf in filter_banks
    ]
    return data, columns


def measure_stats(sig):
    sig_pwr = np.power(sig, 2)
    pwr_max = np.max(sig_pwr)
    var = np.var(sig)
    pwr_q25, pwr_q50, pwr_q75 = \
        np.quantile(sig_pwr, [0.25, 0.5, 0.75])

    columns = [
        'pwr_max', 'var', 'pwr_q25', 'pwr_q50', 'qr_q75'
    ]

    data = [
        pwr_max, var, pwr_q25, pwr_q50, pwr_q75
    ]

    return data, columns


def analyze_signal(signal, filters, fcenters, fs):

    trimmed = trim_signal(signal, fs)

    signal = normalize_energy(trimmed, fs)

    energies, energy_cols = measure_energy_bands(signal, filters, fcenters, fs)
    stats, stats_cols = measure_stats(signal)

    for label, stat in zip(stats_cols, stats):
        print('%s: '%label, stat)

    ax1 = plt.subplot(321)
    ax1.plot(signal)
    ax2 = plt.subplot(322)
    ax2.plot(*magnitude(signal, fs))

    ax3 =plt.subplot(323)
    for bpf in filters:
        ax3.plot(*magnitude(bpf, fs, pad_to=1024))

    ax3.set_ylim((-60, 10))
    ax3.set_xlim((0, fcenters[-1]))

    ax4 = plt.subplot(324)
    ax4.plot(trimmed)

    ax5 = plt.subplot(325)
    ax5.stem(energy_cols, fcenters, energies, use_line_collection=True)

    plt.show()


def process_directory(src_dir, fs):
    output_dir = os.path.join(src_dir, 'Features')
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    for f in os.listdir(dir):
        print('processing ', f)

        if '.npy' not in f:
            print('skipping ', f)
            continue

        src_file = os.path.join(src_dir, f)

        out_file = os.path.join(
            output_dir,
            os.path.splitext(f)[0] + '.npy'
        )

        header_written = False

        with open(out_file, 'w') as f:
            signal, label = load_data(src_file)
            filters, fcenters = generate_filters(100, 20, fs, numtaps=512)

            signal = normalize_energy(
                trim_signal(signal, fs),
                fs
            )

            energies, energy_cols = measure_energy_bands(
                signal,
                filters,
                fcenters,
                fs
            )

            stats, stats_cols = measure_stats(signal)

            if not header_written:
                header_written = True
                header_filename = os.path.join(
                    output_dir,
                    'headers.txt'
                )

                cols = energy_cols
                cols.extend(stats_cols)
                cols.append('label')

                with open(header_filename, 'w') as header_file:
                    header_file.write(
                        ','.join(cols)
                    )

            data = energies
            data.extend(stats)
            data.append(label)

            np.save(out_file, data)


if __name__ == '__main__':
    args = init_args()

    print(args)

    fs = 8000

    if args['dir']:
        # load directory
        dir = args['dir']
        process_directory(dir, fs)


    if args['file']:
        # load file
        signal, label = load_data(args['file'])
        filters, fcenters = generate_filters(100, 10, fs, numtaps=512)
        analyze_signal(signal, filters, fcenters, fs)
