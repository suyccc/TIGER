import soundfile as sf
import numpy as np
import random

def read_wav(path):
    data, samplerate = sf.read(path)
    return data

def write_wav(path, data, samplerate):
    sf.write(path, data, samplerate)
    return

def compute_mch_rms_dB(mch_wav, fs=16000, energy_thresh=-50):
    """Return the wav RMS calculated only in the active portions"""
    mean_square = max(1e-20, np.mean(mch_wav ** 2))
    return 10 * np.log10(mean_square)

if __name__ == '__main__':
    # read all files in a folder
    import os
    from tqdm import tqdm
    total_items = sum([len(dirs) + len(files) for root, dirs, files in os.walk('/mnt/d/img85/data3/LibriSpeech_Habitat/Mono_LibriSpeech_Habitat')])
    progress_bar = tqdm(total=total_items, desc="Searching", ncols=100)
    for root, dirs, files in os.walk('/mnt/d/img85/data3/LibriSpeech_Habitat/Mono_LibriSpeech_Habitat'):
        if "mix.wav" in files:
            spk1_reverb = read_wav(os.path.join(root, "spk1_reverb.wav"))
            spk2_reverb = read_wav(os.path.join(root, "spk2_reverb.wav"))
            noise = read_wav(os.path.join(root, "noise1_reverb.wav"))
            target_refch_energy = compute_mch_rms_dB(spk1_reverb)
            noise_refch_energy = compute_mch_rms_dB(noise)
            intf_refch_energy = compute_mch_rms_dB(spk2_reverb)
            
            sir = random.uniform(-6, 6)
            snr = random.uniform(10, 20)
            
            noise_gain = min(target_refch_energy - noise_refch_energy - snr, 40)
            spk2_gain = min(target_refch_energy - intf_refch_energy - sir, 40)
            
            spk2_reverb = spk2_reverb * 10.**(spk2_gain/20.)
            noise = noise * 10.**(noise_gain/20.)
            
            mix = spk1_reverb + spk2_reverb + noise
            
            os.makedirs(root.replace('img85/data3/LibriSpeech_Habitat/Mono_LibriSpeech_Habitat', 'Mono_LibriSpeech_Habitat_ReMix'), exist_ok=True)
            write_wav(os.path.join(root.replace('img85/data3/LibriSpeech_Habitat/Mono_LibriSpeech_Habitat', 'Mono_LibriSpeech_Habitat_ReMix'), "mix.wav"), mix, 16000)
            write_wav(os.path.join(root.replace('img85/data3/LibriSpeech_Habitat/Mono_LibriSpeech_Habitat', 'Mono_LibriSpeech_Habitat_ReMix'), "spk1_reverb.wav"), spk1_reverb, 16000)
            write_wav(os.path.join(root.replace('img85/data3/LibriSpeech_Habitat/Mono_LibriSpeech_Habitat', 'Mono_LibriSpeech_Habitat_ReMix'), "spk2_reverb.wav"), spk2_reverb, 16000)
            
            progress_bar.update(len(dirs) + len(files))
            