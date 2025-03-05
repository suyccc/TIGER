import os
import random
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
def compute_mch_rms_dB(mch_wav, fs=16000, energy_thresh=-50):
    """Return the wav RMS calculated only in the active portions"""
    mean_square = max(1e-20, torch.mean(mch_wav ** 2))
    return 10 * torch.log10(mean_square)
RATE = 16000
def extend_noise(noise, max_length):
    """ Concatenate noise using hanning window"""
    noise_ex = noise
    window = np.hanning(RATE + 1)
    # Increasing window
    i_w = window[:len(window) // 2 + 1]
    # Decreasing window
    d_w = window[len(window) // 2::-1]
    # Extend until max_length is reached
    while len(noise_ex) < max_length:
        noise_ex = np.concatenate((noise_ex[:len(noise_ex) - len(d_w)],
                                   np.multiply(
                                       noise_ex[len(noise_ex) - len(d_w):],
                                       d_w) + np.multiply(
                                       noise[:len(i_w)], i_w),
                                   noise[len(i_w):]))
    noise_ex = noise_ex[:max_length]
    return noise_ex

def mix_and_save_audio(speech_dir: str, output_dir: str, sample_rate: int = 16000, duration: float = 4.0):
    """
    Given a directory with s1, s2, s3, and noise subdirectories, this function mixes 2 or 3 speakers' audio and saves the mixed audio and individual source audio to the output directory.
    
    Args:
        speech_dir (str): The directory containing s1, s2, s3, and noise subdirectories.
        output_dir (str): The directory to save the mixed and source audio.
        sample_rate (int): The sample rate for the audio (default: 16000).
        duration (float): The duration of the clips to mix (default: 4.0 seconds).
        is_mono (bool): Whether to convert audio to mono (default: True).
    """
    
    # Check and create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    s_list = ['s1', 's2', 's3']
    s1_files = os.listdir(os.path.join(speech_dir, 's1'))
    noise_files = os.listdir(os.path.join(speech_dir, 'noise'))
    speaker_count = {2: 0, 3: 0}
    # Traverse all files in the s1 directory
    for selected_file in tqdm(s1_files):
        # Extract speaker names from the filename (assuming the format: speaker1_speaker2_speaker3.wav)
        speaker1, speaker2, speaker3 = selected_file.replace('.wav', '').split('_')
        
        # Create a list of speakers
        speakers = [speaker1, speaker2, speaker3]
        
        # Randomly choose 2 or 3 speakers
        num_spks = random.choice([2, 3])
        
        # Load the corresponding speaker audio files
        speaker_wavs = []
        for i in range(num_spks):
            speaker_wav, _ = torchaudio.load(os.path.join(speech_dir, s_list[i], f'{speaker1}_{speaker2}_{speaker3}.wav'))
            speaker_wavs.append(speaker_wav.squeeze())

        # Load noise audio
        noise_wav, _ = torchaudio.load(os.path.join(speech_dir, 'noise', random.choice(noise_files)))
        noise_wav = noise_wav.squeeze()
        # # Random start and end positions for the segment
        # start = random.randint(0, speaker_wavs[0].shape[-1] - sample_rate * duration)
        # end = int(start + sample_rate * duration)

        # # Extract a segment from the audio files
        # selected_speakers = [wav[..., start:end] for wav in speaker_wavs]
        # noise_segment = noise_wav[..., start:end]
        selected_speakers = speaker_wavs
        noise_segment = noise_wav
        # --- SIR Adjustment ---
        # Random SIR for each speaker
        sirs = torch.Tensor(num_spks-1).uniform_(-6,6).numpy()
        target_refch_energy = compute_mch_rms_dB(selected_speakers[0])

        for i in range(num_spks-1):
            sir = sirs[i]
            intf_refch_energy = compute_mch_rms_dB(selected_speakers[i+1])
            gain = min(target_refch_energy - intf_refch_energy - sir, 40)
            selected_speakers[i + 1] *= 10. ** (gain / 20.)

        # --- SNR Adjustment ---
        all_speech = torch.sum(torch.stack(selected_speakers), dim=0)
        extended_noise = extend_noise(noise_segment.numpy(), all_speech.shape[-1])
        all_noise = torch.from_numpy(extended_noise)
        # import pdb; pdb.set_trace()
        
        target_refch_energy = compute_mch_rms_dB(all_speech)
        snr = torch.Tensor(1).uniform_(10, 20).numpy()
        noise_refch_energy = compute_mch_rms_dB(all_noise)
        gain = min(target_refch_energy - noise_refch_energy - snr, 40)
        all_noise *= 10. ** (gain / 20.)

        # Mix the speech and noise
        mixed_wav = all_speech + all_noise
        if num_spks == 2:
            mixed_name = f"{speaker1}_{speaker2}.wav"
        else:
            mixed_name = f"{speaker1}_{speaker2}_{speaker3}.wav"
        # Save the mixed audio and source audio
        mix_path = os.path.join(output_dir, 'remix', mixed_name)
        os.makedirs(os.path.join(output_dir, 'remix'), exist_ok=True)

        torchaudio.save(mix_path, mixed_wav.unsqueeze(0), sample_rate)
        
        # Save individual sources
        for j, speaker_wav in enumerate(selected_speakers[:num_spks]):
            speaker_path = os.path.join(output_dir, f's{j+1}', mixed_name)
            os.makedirs(os.path.join(output_dir, f's{j+1}'), exist_ok=True)
            torchaudio.save(speaker_path, speaker_wav.unsqueeze(0), sample_rate)
        # Save noise
        noise_path = os.path.join(output_dir, 'noise', mixed_name)
        os.makedirs(os.path.join(output_dir, 'noise'), exist_ok=True)
        torchaudio.save(noise_path, all_noise.unsqueeze(0), sample_rate)
        speaker_count[num_spks] += 1
    print(f"Generated mixed audio files for all {len(s1_files)} samples to {output_dir}.")
    print(f"2 speakers: {speaker_count[2]}, 3 speakers: {speaker_count[3]}")
    
if __name__ == '__main__':
    speech_dir = '/gpfs-flash/hulab/public_datasets/audio_datasets/librispeech/Libri3Mix/wav16k/min/train-100'
    output_dir = '/gpfs-flash/hulab/suyuchang/LibriMix/train-100'
    # fix random seed
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    mix_and_save_audio(speech_dir, output_dir)