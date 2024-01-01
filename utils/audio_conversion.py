from subprocess import Popen, STDOUT
from os import listdir, devnull
from tqdm import tqdm

class convert_media():
    audio_sr = 8000 # sampling rate for audio
    FNULL = open(devnull, "w")

    def __init__(self):
        pass
    def media_convertor(self, src_file, dst_file, sample_rate=8000):
        cmd = ["ffmpeg", "-i",  src_file, "-b:a", f"{sample_rate}", dst_file]
        return self.run_command(cmd)

    def run_command(self, command):
        """Run a command, given an array of the command and arguments"""
        proc = Popen(command, shell=False, stdout=self.FNULL, stderr=STDOUT)
        ret = proc.wait()
        if ret != 0:
            print(f"Conversion failed: {command[2]}")
            return False
        return True
    
    def batch_wav_convert(self, src, dst, sample_rate=8000, description="Converting audio"):
        files = listdir(src)
        for file in tqdm(files, desc=description):
            f_src = f"{src}/{file}"
            n_filename = ''.join(file.split('.')[:-1])
            f_dst = f"{dst}/{n_filename}.wav"
            self.media_convertor(f_src, f_dst, sample_rate=sample_rate)
        return

if __name__ == "__main__":
    main = convert_media()
    srcfld = "data/dataset/keyword_raw"
    dstfld = "data/dataset/keyword"

    main.batch_wav_convert(srcfld, dstfld)