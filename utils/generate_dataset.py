import os, shutil, librosa, random, soundfile
from tqdm import tqdm
try:
    from audio_conversion import convert_media
    from audio_filters import audio_filters
    data_src = "../data/dataset/"
except:
    from utils.audio_conversion import convert_media
    from utils.audio_filters import audio_filters
    data_src = "data/dataset/"
    
    
class dataset_generator:
    # constants
    audio_length = 1 # 1 second
    sample_rate = 8000 # 8khz
    ambient_size = 2000 # size of ambient sounds
    wake_word_size = 1400 # size of wake-word sounds
    audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.aiff', '.ra', '.mid', '.midi']
    temp_dir = 'temp/'
    train_test_ratio = 0.8 # 80% train and 20% test
    
    # start
    def __init__(self) -> None:
        self.initialize()
        
    def initialize(self):
        self.Filters = audio_filters()
        self.Convertor = convert_media()
        
    def generate_dataset(self, reset=True):
        if not reset:
            check = self.check_dataset_presence()
            if check:
                return True
        # load ambient sounds
        ambient_sounds = self.load_sounds(f"{data_src}ambient-sounds", description="loading ambient sounds:")
        # generate ambient sound dataset
        total_ambients = self.generate_ambient_data(ambient_sounds)
        # load wake word sounds
        wake_sounds = self.load_sounds(f"{data_src}wake-word", description="loading wake sounds:")
        # generate wake sound dataset
        self.generate_wake_word_data(wake_sounds, ambient_sounds, total_ambients)
        # check dataset presence
        return self.check_dataset_presence()
        
    def generate_wake_word_data(self, wake_sounds:list, ambient_sounds, total_ambients):
        N_iterations =  max(1, total_ambients//len(wake_sounds)) # generate k iteration of each sounds
        data = []
        for wav in tqdm(wake_sounds, desc="generating wake audio:"):
            waves = self.Filters.filter_audio(wav, ambient_sounds, N_iterations, self.audio_length, self.sample_rate)
            data.extend(waves)
        # shuffle audio
        random.shuffle(data)
        train_len = int(len(data)*self.train_test_ratio)
        train_data, test_data = data[:train_len], data[train_len:]
        # generating audio files
        folder_dst_train = f"{data_src}sounds/train/wake_word/"
        folder_dst_test = f"{data_src}sounds/val/wake_word/"
        os.makedirs(os.path.dirname(folder_dst_train), exist_ok=True)
        os.makedirs(os.path.dirname(folder_dst_test), exist_ok=True)
        # storing train audio
        count = 1
        for wav in tqdm(train_data, desc="storing wake train audio:"):
            dst = f"{folder_dst_train}audio{count}.wav"
            try:
                soundfile.write(dst, wav, self.sample_rate)
                count += 1
            except:
                pass
        # storing validation audio
        count = 1
        for wav in tqdm(test_data, desc="storing wake val audio:"):
            dst = f"{folder_dst_test}audio{count}.wav"
            try:
                soundfile.write(dst, wav, self.sample_rate)
                count += 1
            except:
                pass
        return
        
    def generate_ambient_data(self, ambient_sounds:list):
        sample_k = max(1, self.ambient_size//len(ambient_sounds)) # extract k samples per sounds
        wav_len = int(self.sample_rate*self.audio_length)
        data = []
        for wav in tqdm(ambient_sounds, desc="extracting ambient audio:"):
            len_ = len(wav)
            space = len_ - wav_len
            if space < 0: space = 0
            for i in range(sample_k):
                start = random.randint(0, space)
                data.append(wav[start:start+wav_len])
        # extract N audios
        N_extractions = min(len(data), self.ambient_size)
        data = random.sample(data, N_extractions)
        # shuffle audio
        random.shuffle(data)
        train_len = int(len(data)*self.train_test_ratio)
        train_data, test_data = data[:train_len], data[train_len:]
        # generating audio files
        folder_dst_train = f"{data_src}sounds/train/ambient_sound/"
        folder_dst_test = f"{data_src}sounds/val/ambient_sound/"
        os.makedirs(os.path.dirname(folder_dst_train), exist_ok=True)
        os.makedirs(os.path.dirname(folder_dst_test), exist_ok=True)
        # storing train audio
        count = 1
        for wav in tqdm(train_data, desc="storing ambient train audio:"):
            dst = f"{folder_dst_train}audio{count}.wav"
            try:
                soundfile.write(dst, wav, self.sample_rate)
                count += 1
            except:
                pass
        # storing validation audio
        count = 1
        for wav in tqdm(test_data, desc="storing ambient val audio:"):
            dst = f"{folder_dst_test}audio{count}.wav"
            try:
                soundfile.write(dst, wav, self.sample_rate)
                count += 1
            except:
                pass
        return N_extractions
    
    # utils
    def check_dataset_presence(self):
        try:
            # ambient sounds
            train_set = len([x for x in os.listdir(f"{data_src}sounds/train/ambient_sound/") if x.endswith('.wav')])
            val_set = len([x for x in os.listdir(f"{data_src}sounds/val/ambient_sound/") if x.endswith('.wav')])
            # wake word
            train_set_ = len([x for x in os.listdir(f"{data_src}sounds/train/wake_word/") if x.endswith('.wav')])
            val_set_ = len([x for x in os.listdir(f"{data_src}sounds/val/wake_word/") if x.endswith('.wav')])
            print(f"Dataset checked, train: {train_set_}/{train_set}, val: {val_set_}/{val_set}")
            if train_set+train_set_>200 and val_set+val_set_>40:
                return True
            return False
        except Exception as e:
            print(f"Failed to check dataset existence: {e}")
            return False
        
    def load_sounds(self, folder_path:str, description="loading:"):
        os.makedirs(os.path.dirname(self.temp_dir), exist_ok=True)
        audios = []
        for file_name in tqdm(os.listdir(folder_path), desc=description):
            if self.is_file_allowed(file_name):
                file_src = f"{folder_path}/{file_name}"
                if not file_name.endswith(".wav"):
                    name = ''.join(file_name.split('.')[:-1])
                    file_dst = f"{self.temp_dir}{name}.wav"
                    status = self.Convertor.media_convertor(file_src, file_dst, sample_rate=self.sample_rate)
                    file_src = file_dst if status else ""
                wav = self.load_audio(file_src)
                if type(wav)!=bool:
                    audios.append(wav)
                
        # Delete the directory
        shutil.rmtree(os.path.dirname(self.temp_dir), ignore_errors=True)
        return audios
    
    def load_audio(self, filepath:str):
        try:
            wav, _ = librosa.load(filepath, mono=True, sr=self.sample_rate)
            return wav
        except:
            return False
        
        
    def is_file_allowed(self, src:str):
        extension = '.' + src.split('.')[-1]
        if extension in self.audio_extensions:
            return True
        return False
    
    