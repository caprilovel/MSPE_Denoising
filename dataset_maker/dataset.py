#%%
import numpy
import torch


import wfdb

data_path = '/local/storage/ding/mit-bih-arrhythmia-database-1.0.0/'
noise_path = '/local/storage/ding/mit-bih-noise-stress-test-database-1.0.0/'
data = wfdb.rdrecord(data_path + '118', physical=True)
noise_data = wfdb.rdrecord(noise_path + 'em', physical=True)
signal = data.p_signal
noise = noise_data.p_signal
print(signal.shape)
print(noise.shape)

#%%
import matplotlib.pyplot as plt
plt.plot(signal[:512])

#%%
def generate_noisy_data(signal, noise, snr):
    signal_power = numpy.sum(signal ** 2)
    noise_power = numpy.sum(noise ** 2)
    noise = noise * numpy.sqrt(signal_power / (noise_power * snr))
    return signal + noise

def normalize(signal):
    return (signal - numpy.mean(signal)) / numpy.std(signal)


noisy_signal_10 = generate_noisy_data(signal[:512, 0], noise[:512, 0], 10)
noisy_signal_20 = generate_noisy_data(signal[:512, 0], noise[:512, 0], 20)
noisy_signal_30 = generate_noisy_data(signal[:512, 0], noise[:512, 0], 30)

plt.plot(noisy_signal_10)
plt.plot(noisy_signal_20)
plt.plot(noisy_signal_30)
#%%
noisy_signal_18 = wfdb.rdrecord(noise_path + '118e18', physical=True).p_signal[140000:141000, 0]
noisy_signal_6 = wfdb.rdrecord(noise_path + '118e06', physical=True).p_signal[140000:141000, 0]
plt.plot(noisy_signal_18)
plt.plot(noisy_signal_6)
# plt.plot(signal[:512, 0])
plt.legend(['18', '12'])