from librosa import effects
from scipy.signal import lfilter, butter
import numpy as np
from random import choice, randint

def change_pitch_speed(samples, sample_rate=8000):
    y_pitch_speed = samples.copy()
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high = 1)
    speed_fac = 1.0  / length_change
    tmp = np.interp(np.arange(0,len(y_pitch_speed),speed_fac),np.arange(0,len(y_pitch_speed)),y_pitch_speed)
    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed *= 0
    y_pitch_speed[0:minlen] = tmp[0:minlen]
    return y_pitch_speed

def change_pitch(samples, sample_rate=8000):
    y_pitch = samples.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    #print("pitch_change = ",pitch_change)
    y_pitch = effects.pitch_shift(y_pitch.astype('float64'), 
                                          sample_rate, n_steps=pitch_change, 
                                          bins_per_octave=bins_per_octave)
    return y_pitch

def change_speed(samples, sample_rate=8000):
    # handle both increase and decrease randomly
    y_speed = samples.copy()
    speed_change = np.random.uniform(low=0.9,high=1.1)
    tmp = effects.time_stretch(y_speed.astype('float64'), speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed *= 0 
    y_speed[0:minlen] = tmp[0:minlen]
    return y_speed

def change_value_augment(samples, sample_rate=8000):
    # like increasing volumen randomly
    y_aug = samples.copy()
    dyn_change = np.random.uniform(low=1.5,high=3)
    y_aug = y_aug * dyn_change
    return y_aug

def change_add_dist_noise(samples, sample_rate=8000):
    y_noise = samples.copy()
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.008*np.random.uniform()*np.amax(y_noise)
    y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])
    return y_noise

def change_shift(samples, sample_rate=8000):
    y_shift = samples.copy()
    timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length
    #print("timeshift_fac = ",timeshift_fac)
    start = int(y_shift.shape[0] * timeshift_fac)
    #print(start)
    if (start > 0):
        y_shift = np.pad(y_shift,(start,0),mode='constant')[0:y_shift.shape[0]]
    else:
        y_shift = np.pad(y_shift,(0,-start),mode='constant')[0:y_shift.shape[0]]
    return y_shift

def change_streching(samples, sample_rate=8000):
    input_length = len(samples)
    streching = samples.copy()
    streching = effects.time_stretch(streching.astype('float'), 1.1)
    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
    return streching

def butter_params(low_freq, high_freq, fs, order=5):
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, low_freq, high_freq, fs, order=5):
    b, a = butter_params(low_freq, high_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y
def telephone_effect(samples, sample_rate=8000):
    fs,audio = sample_rate, samples.copy()
    low_freq = choice([250.0, 300.0, 350.0])
    high_freq = 3000.0
    order_ = choice([3,4,5]) #6
    filtered_signal = butter_bandpass_filter(audio, low_freq, high_freq, fs, order=order_)
    
    return filtered_signal

def echo_effect(samples, sample_rate=8000):
    _,audio = sample_rate, samples.copy()
    delay_ = 0.1#choice([0.2,0.5,0.6]) # in sec
    wav = audio.copy()
    repeat = 2#choice([1,2,3,4])
    echo_volume = 1#choice([0.5,1,1.6,1.8])
    for rep in range(repeat):
        audio = audio*(echo_volume**(rep+1))
        d_idx = int(sample_rate*delay_)
        delay_audio = [0 for _ in range(d_idx)]
        delay_audio.extend(audio.tolist()[d_idx:])
        delay_audio = np.array(delay_audio)
        wav = wav + delay_audio
    return wav


class audio_filters:
    def __init__(self):
        self.functions = [change_pitch_speed, change_pitch, change_speed,change_value_augment,
                    change_add_dist_noise, change_shift, change_streching]
        self.functions_c = [[change_pitch_speed, change_value_augment,change_add_dist_noise],
                    [change_add_dist_noise,change_value_augment,change_shift],
                    [change_shift, change_add_dist_noise,change_streching]]
        self.cell_effect_functions = [telephone_effect, echo_effect]
        
    def filter_audio(self, wav, waves:list, N_iterations:int, audio_length=1, sample_rate=8000):
        # take a single wav and ambients waves then return multiple iterations
        req_wav_len = int(sample_rate*audio_length)
        wav = self.get_audio_chunck(wav, req_wav_len)
        data = [wav] # one down N to go
        for i in range(N_iterations-1):
            # roll the dice
            dice = randint(0, 10)
            rand_ambient_wav = choice(waves)
            wav_out = self.merge_wav(wav, rand_ambient_wav)
            # 0-2 simple merge
            if dice<=2:
                pass
            # 3-4 merge and single filter
            elif dice<=4:
                wav_out = choice(self.functions)(wav_out, sample_rate)
            # 5-6 merge and cell filter
            elif dice<=6:
                wav_out = choice(self.cell_effect_functions)(wav_out, sample_rate)
            # 7-8 merge and multi filter
            elif dice<=8:
                for fnc in choice(self.functions_c):
                    wav_out = fnc(wav_out, sample_rate)
            # 9-10 merge and simple filter and cell filter
            else:
                wav_out = choice(self.functions)(wav_out, sample_rate)
                wav_out = choice(self.cell_effect_functions)(wav_out, sample_rate)
            data.append(wav_out)
        return data
            
    def get_audio_chunck(self, wav, wav_len:int):
        # open
        len_ = len(wav)
        space = len_ - wav_len
        if space < 0: space = 0
        start = randint(0, space)
        wav_out = wav[start:start+wav_len]
        len_ = len(wav_out)
        if len_<wav_len: # complete it
            wav = np.full((wav_len), 0, dtype=np.float32)
            wav[0:len_] = wav_out
            wav_out = wav
        return wav_out
    
    def merge_wav(self, wav, wav_n):
        wav_p = wav.copy()
        wav_neg = wav_n.copy()
        l1 = len(wav_p)
        l2 = len(wav_neg)
        space = l1 - l2
        pos_space = randint(0, abs(space))
        scale = choice([0.1, 0.14, 0.18, 0.2])
        if space > 0:
            wav_p[pos_space:l2 + pos_space] += wav_neg*scale
            out = wav_p
        else:
            wav_neg = wav_neg*scale
            wav_neg[pos_space:l1 + pos_space] += wav_p
            out = wav_neg
        return out
        