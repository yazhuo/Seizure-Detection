from glob import glob
import os
import numpy as np
import scipy
from scipy.signal import find_peaks
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft
from scipy.signal import welch
from collections import OrderedDict, Counter
# from detecta import detect_peaks
from detect_peaks import detect_peaks
import pandas as pd
from scipy.signal import kaiserord, lfilter, firwin, freqz
import pywt

# DATA_SRC_PATH = "../../../data/"
DATA_SRC_PATH = "../../DataSet/"
CLASSES = ["ictal", "interictal"]  # [0, 1]

ANIMAL_NAME = "Patient"
CLASS_NAME = "ictal"
TOP_K_PEAKS = 2
DO_PLOT = True
DO_DISABLE = True
DO_DEBUG = False
WAVELET_NAME = "haar" # Harr --> haar; Morlet wavelet --> morl; Daubechies --> db
WAVELET_NUM_LEVELS = 4 # max: 7; but stick with 4
N_CHANNELS = 16
LOW_PASS_FREQ = 60
PEAK_FINDING_PERCENTILE_THRESHOLD = 5
N_PATIENTS = range(1, 9)


def load_file_scan(patient_idx, animal="Patient", class_name="ictal"):
    regex_files = os.path.join(DATA_SRC_PATH, "%s_%d" % (animal, patient_idx), "%s_%d_%s_segment_*.mat" % (animal, patient_idx, class_name))
    files = glob(regex_files)
    return files


def get_psd_values(y_values, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    if DO_PLOT and not DO_DISABLE:
        plt.figure(1)
        plt.plot(f_values, psd_values, linestyle='-', color='blue')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2 / Hz]')
        plt.savefig("PSD_value_of_signal.png")
        plt.show()
    return f_values, psd_values


def signal_detrending(y):
    ylin = signal.detrend(y, type='linear')

    if DO_DEBUG:
        yconst = signal.detrend(y, type='constant')
    if DO_PLOT and not DO_DISABLE:
        plt.figure(2)
        t = np.linspace(0, 1, y.size)
        plt.plot(t, y, '-rx', 1)
        # plt.plot(t, yconst, '-bo', 1)
        plt.plot(t, ylin, '-k+', 1)
        plt.grid()
        plt.legend(['signal', 'const. detrend', 'linear detrend'])
        plt.savefig("detrended_signal.png")
        plt.show()
    # TODO: only apple linear detrend
    return ylin


def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[0:N // 2])
    return f_values, fft_values


def get_FFT(signal, freq):
    N = signal.size
    t_n = 1
    T = t_n / N
    f_s = 1 / T
    f_values, fft_values = get_fft_values(signal, T, N, f_s)
    if DO_PLOT and not DO_DISABLE:
        plt.plot(f_values, fft_values, linestyle='-', color='blue')
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.title("Frequency domain of the signal", fontsize=16)
        plt.show()
    return f_values, fft_values


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2:]


def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values


def LP_filter_preprocess_signal(signal_data, T, num_samples, f_s):
    nyq_rate = f_s / 2.0

    # The desired width of the transition from pass to stop, relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    # width = 5.0 / nyq_rate # FIXME: large the (5.) distortion span is less !
    width = 100.0 / nyq_rate  # GOOD; but smoothing is higher
    # width = 20.0 / nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    # The cutoff frequency of the filter.
    cutoff_hz = 60.0
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))
    # Use lfilter to filter x with the FIR filter.
    filtered_x = lfilter(taps, 1.0, signal_data)
    # ------------------------------------------------
    # Plot the FIR filter coefficients.
    # ------------------------------------------------
    if DO_PLOT and not DO_DISABLE:
        plt.figure(3)
        plt.plot(taps, 'bo-', linewidth=2)
        plt.title('Filter Coefficients (%d taps)' % N)
        plt.grid(True)
        plt.savefig("FIR_filter_coefficients.png")
    # ------------------------------------------------
    # Plot the magnitude response of the filter.
    # ------------------------------------------------
    w, h = freqz(taps, worN=8000)
    if DO_PLOT and not DO_DISABLE:
        plt.figure(4)
        plt.clf()
        plt.plot((w / np.pi) * nyq_rate, np.absolute(h), linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Frequency Response')
        plt.ylim(-0.05, 1.05)
        plt.grid(True)

    # Upper inset plot.
    if DO_PLOT and not DO_DISABLE:
        ax1 = plt.axes([0.42, 0.6, .45, .25])
        plt.plot((w / np.pi) * nyq_rate, np.absolute(h), linewidth=2)
        plt.xlim(0, 8.0)
        plt.ylim(0.9985, 1.001)
        plt.grid(True)

    # Lower inset plot
    if DO_PLOT and not DO_DISABLE:
        ax2 = plt.axes([0.42, 0.25, .45, .25])
        plt.plot((w / np.pi) * nyq_rate, np.absolute(h), linewidth=2)
        plt.xlim(12.0, 20.0)
        plt.ylim(0.0, 0.0025)
        plt.grid(True)
        plt.savefig("FIR_filter_magnitude_response.png")
    # ------------------------------------------------
    # Plot the original and filtered signals.
    # ------------------------------------------------
    # The phase delay of the filtered signal.
    delay = 0.5 * (N - 1) / f_s

    t = np.linspace(0, 1, num_samples)
    front_pad = np.zeros(int(np.floor((N - 1) / 2)))
    back_pad = np.zeros(int(np.ceil((N - 1) / 2)))
    processed_sig = np.append(np.append(front_pad, filtered_x[N - 1:]), back_pad)
    if DO_PLOT:
        plt.figure(5)
        # Plot the original signal.
        plt.plot(t, signal_data)
        # Plot the filtered signal, shifted to compensate for the phase delay.
        # plt.plot(t - delay, filtered_x, 'r-') # FIXME:
        # Plot just the "good" part of the filtered signal.  The first N-1
        # samples are "corrupted" by the initial conditions.
        # plt.plot(t[N - 1:] - delay, filtered_x[N - 1:], 'g', linewidth=2) # FIXME:
        plt.xlabel('t')
        plt.grid(True)
        plt.plot(t, processed_sig, 'g', linewidth=2)
        plt.savefig("FIR_LP_filtered_signal.png")
        plt.show()

    return processed_sig, T, N, f_s


def wavelet_apply_wavedec_and_waverec(signal_data, wavelet_name="morl", num_level=4):
    # print(pywt.wavelist(kind='discrete'))
    coeffs = pywt.wavedec(signal_data, wavelet_name, level=num_level)
    if DO_PLOT and not DO_DISABLE:
        reconstructed_signal = pywt.waverec(coeffs, wavelet_name)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(signal_data, label='signal')
        ax.plot(reconstructed_signal, label='reconstructed signal', linestyle='--')
        ax.legend(loc='upper left')
        ax.set_title('de- and reconstruction using wavedec()')
        plt.savefig("wavelet_wavedec_and_waverec_TD_signals.png")
        plt.show()
    return coeffs


# ======================================================================================================================
# ======================= Some useful statistical features and other signal prcessing features =========================
# ======================================================================================================================
def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy


def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    # std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values ** 2))
    # return [n5, n25, n75, n95, median, mean, std, var, rms]
    return [n5, n25, n75, n95, median, mean, var, rms]


def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


# set of 12 features for any list of values
def get_statistical_and_other_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics


# ======================================================================================================================
# ========================================= End of above feature calculation ===========================================
# ======================================================================================================================


def extract_and_concat_features(signal_data_file):
    # LOW_PASS_FREQ TODO
    def get_first_n_peaks(x, y, no_peaks=TOP_K_PEAKS):
        x_, y_ = list(x), list(y)
        if len(x_) > no_peaks:
            return x_[:no_peaks], y_[:no_peaks]
        else:
            missing_no_peaks = no_peaks - len(x_)
            return x_ + [0] * missing_no_peaks, y_ + [0] * missing_no_peaks

    def get_time_freq_domain_features(x_values, y_values, mph):
        x_values_low_pass = [] # updated values
        y_values_low_pass = [] # updated values
        for x, y in zip(x_values, y_values):
            if x <= LOW_PASS_FREQ:
                x_values_low_pass.append(x)
                y_values_low_pass.append(y)
        x_values_low_pass = np.asarray(x_values_low_pass)
        y_values_low_pass = np.asarray(y_values_low_pass)
        indices_peaks = detect_peaks(y_values_low_pass, mph=mph)
        peaks_x, peaks_y = get_first_n_peaks(x_values_low_pass[indices_peaks], y_values_low_pass[indices_peaks])
        # print(peaks_y) # TODO: remove
        return peaks_y # don't pass peaks_x because there's no use of it

    def extract_features_labels(dataset, T, N, f_s, denominator):
        features = []
        for signal_no in range(0, len(dataset)):

            signal_data = dataset[signal_no, :]

            # ====================== Preprocess signal ===================
            # No need to apply low-pass filtering, since we are only going to capture signal below 60Hz
            # NOTE: FIXME : DC component removal: detrend
            signal_ready = signal_detrending(signal_data)
            # =============================================================

            # =========== Extract Time & Freq domain features =============
            signal_min = np.nanpercentile(signal_ready, PEAK_FINDING_PERCENTILE_THRESHOLD)
            signal_max = np.nanpercentile(signal_ready, 100 - PEAK_FINDING_PERCENTILE_THRESHOLD)
            # ijk = (100 - 2*PEAK_FINDING_PERCENTILE_THRESHOLD)/10
            mph = signal_min + (signal_max - signal_min) / denominator

            features += get_time_freq_domain_features(*get_psd_values(signal_ready, f_s), mph)
            # Feature headers : FIXME: ["PSD_peaks_y-1_ch-1", "PSD_peaks_y_2_ch_1"] # 2
            features += get_time_freq_domain_features(*get_fft_values(signal_ready, T, N, f_s), mph)
            # Feature headers : FIXME: ["FFT_peaks_y_1_ch_1", .., "FFT_peaks_y_2_ch_1"] # 2
            features += get_time_freq_domain_features(*get_autocorr_values(signal_ready, T, N, f_s), mph)
            # Feature headers : FIXME: ["AutoCorr_peaks_y_1_ch_1", "AutoCorr_peaks_y_2_ch_1"] # 10
            #  =============================================================

            # ================= Extract Wavelet features =================
            wavelet_coeffs = wavelet_apply_wavedec_and_waverec(signal_data, wavelet_name=WAVELET_NAME,
                                                               num_level=WAVELET_NUM_LEVELS)
            # Above return length of 4
            # [cA_n, cD_n, cD_n - 1, …, cD2, cD1]
            for coeff in wavelet_coeffs[:-1]: # Leave-out high-fre components
                features += get_statistical_and_other_features(coeff)
                # Feature headers : FIXME: ["Entropy_coeff-1_ch-1", "ZeroCrossings_coeff-1_ch-1",
                #  "MeanCrossings_coeff-1_ch-1", "Stats features]

            # =============================================================

        return np.array(features)

    data_dict = loadmat(signal_data_file)
    freq = data_dict['freq']
    data = data_dict['data']
    animal_class = signal_data_file.split("/")[-1].split("_")[2]
    if animal_class == "ictal":
        train_label = [0]
    else:
        train_label = [1]

    N = data.shape[1]
    t_n = 1
    T = t_n / N
    f_s = freq  # 1 / T

    # Use In-Built function for feature extraction
    denominator = 10  # FIXME: tune this parameter
    X_train_sample = extract_features_labels(data, T, N, f_s, denominator)

    # print("Feature extracted size", X_train_sample.shape) # TODO: remove

    return X_train_sample, train_label, data.shape[0]


def main(patient_idx, class_name):
    """
    :return:
    """
    animal_name = ANIMAL_NAME
    # class_name = CLASS_NAME
    mat_files = load_file_scan(patient_idx, animal=animal_name, class_name=class_name)
    # print(mat_files) # TODO: remove

    list_of_X_train_features = []
    list_of_Y_train_labels = []
    list_filenames = []
    n_channels = 16
    for mat_file_idx, mat_file in enumerate(mat_files):
        X_train, Y_train, n_channels = extract_and_concat_features(mat_file)
        list_of_X_train_features.append(X_train)
        list_of_Y_train_labels.append(Y_train)
        list_filenames.append(mat_file.split("/")[-1])

        # if mat_file_idx == 2:
        if mat_file_idx % 100 == 0:
            print("%d files processed..." % (mat_file_idx))
            # break # TODO: remove

        # break # TODO: remove
    X_train_features = np.asarray(list_of_X_train_features)
    Y_train_labels = np.asarray(list_of_Y_train_labels)
    filenames = np.asarray(list_filenames)

    # TODO: remove
    # print(X_train_features.shape)
    # print(Y_train_labels.shape)
    # print(filenames.shape)

    # ==============================> Manual features header design
    # ============> Wavelet domain feature headers
    # statistical_header = ["%5 val", "%25 val", "%75 val", "%95 val", "Median", "Mean", \
    #                       "Std", "Variance", "RMS"]
    statistical_header = ["%5 val", "%25 val", "%75 val", "%95 val", "Median", "Mean", \
                          "Variance", "RMS"]
    other_wavelet_header = ["Entropy", "Num_zero_crossings", "Num_mean_crossings"]
    coeff_wavelet_header = other_wavelet_header + statistical_header
    wavelet_coeff_header = ["cALevel-%d"%(WAVELET_NUM_LEVELS)] + ["cDLevel-%d"%e for e in range(WAVELET_NUM_LEVELS, 1, -1)]
    wavelet_header = ["Coeff-%s " % i + header for header in coeff_wavelet_header for i in
                      wavelet_coeff_header]

    # ============> Time & Freq domain feature headers
    PSD_peaks_header = ["PSD peaks %s-%s" % (coord, i + 1) for coord in ["y"] for i in range(TOP_K_PEAKS)]
    FFT_peaks_header = ["FFT peaks %s-%s" % (coord, i + 1) for coord in ["y"] for i in range(TOP_K_PEAKS)]
    AutoCorr_peaks_header = ["AutoCorr peaks %s-%s" % (coord, i + 1) for coord in ["y"] for i in
                             range(TOP_K_PEAKS)]
    freq_domain_header = PSD_peaks_header + FFT_peaks_header + AutoCorr_peaks_header

    # ============> Composite feature headers
    comb_header = wavelet_header + freq_domain_header

    # ============> Channel-wise feature headers
    expanded_features_header = ["Channel-%d " % i + header for i in range(n_channels) for header in comb_header]
    # print(len(expanded_features_header)) # TODO: remove
    # feat_p = X_train_features.shape[1]
    # df = pd.DataFrame(X_train_features, columns=['feature_%s' % (i+1) for i in range(feat_p)])
    df = pd.DataFrame(X_train_features, columns=expanded_features_header)
    df.insert(0, 'labels', Y_train_labels)
    df.insert(0, 'filenames', filenames)
    df.to_csv("Revised_train_%s_%d_%s_data_v1.csv" % (animal_name, patient_idx, class_name))
    print("Saved Table Shape : ", df.shape)


if __name__ == "__main__":
    for patient_idx in N_PATIENTS:
        for class_name in CLASSES:
            main(patient_idx, class_name)
