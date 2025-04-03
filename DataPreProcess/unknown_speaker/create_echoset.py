import os
import random
import numpy as np
import soundfile as sf
from tqdm import tqdm
from collections import defaultdict

def read_wav(path):
    """读取音频文件并返回 (waveform, samplerate)"""
    data, samplerate = sf.read(path)
    return data, samplerate

def write_wav(path, data, samplerate):
    """写出音频到指定路径"""
    sf.write(path, data, samplerate)

def compute_mch_rms_dB(wav_data):
    """
    计算给定音频的 RMS(dB)。
    如果需要对静音做剔除，可在此处加 VAD 或能量阈值判断。
    """
    mean_square = max(1e-20, np.mean(wav_data**2))
    return 10 * np.log10(mean_square)

def apply_gain(ref_energy, in_wav, ratio_db):
    """
    ref_energy: 参考音频(通常是主说话人)的能量
    in_wav:  需要被增益调整的音频(干扰或噪声)
    ratio_db: 想要达到的 ref_wav 与 in_wav 之间的 dB 差
    """
    in_energy = compute_mch_rms_dB(in_wav)
    # 计算需要的增益(单位 dB)
    gain_db = ref_energy - in_energy - ratio_db
    # import pdb; pdb.set_trace()
    gain_db = min(gain_db, 40)
    scale = 10 ** (gain_db / 20.0)
    return in_wav * scale

def build_speaker_dict(root_dir):
    """
    第一遍：遍历整个数据集，建立一个字典：
    speaker_dict[speaker_id] = [该说话人的所有音频路径列表]

    假设每个子目录名形如 '1089_260'，只有两个说话人 spk1, spk2。
    """
    speaker_dict = defaultdict(list)

    for root, dirs, files in os.walk(root_dir):
        dir_name = os.path.basename(root)
        # 判断目录名里是否含下划线
        if '_' not in dir_name:
            continue

        spk_ids = dir_name.split('_')
        # 这里假设每个目录只有 2 个说话人 ID
        if len(spk_ids) != 2:
            continue

        spk1, spk2 = spk_ids
        spk1_path = os.path.join(root, 'spk1_reverb.wav')
        spk2_path = os.path.join(root, 'spk2_reverb.wav')

        if os.path.exists(spk1_path):
            speaker_dict[spk1].append(spk1_path)
        if os.path.exists(spk2_path):
            speaker_dict[spk2].append(spk2_path)

    return speaker_dict

if __name__ == '__main__':
    split = 'val'
    root_root_dir = f'/gpfs-flash/hulab/public_datasets/audio_datasets/LibriSpeech_Habitat/Mono_LibriSpeech_Habitat/{split}' 
    place_dirs = os.listdir(root_root_dir)
    for place_dir in os.listdir(root_root_dir):
        now_place_dir = os.path.join(root_root_dir, place_dir)
        for room_dir in os.listdir(now_place_dir):
            now_root_dir = os.path.join(now_place_dir, room_dir)
            now_out_root_dir = now_root_dir.replace('public_datasets/audio_datasets/LibriSpeech_Habitat/Mono_LibriSpeech_Habitat', 'suyuchang/EchoSet3')
            # ----------------------
            # 第 1 步：构建 {说话人ID -> [音频路径列表]} 映射
            # ----------------------
            print(f"Building speaker dictionary: {split}/{place_dir}/{room_dir}")
            speaker_dict = build_speaker_dict(now_root_dir)
            all_speakers = list(speaker_dict.keys())
            print(f"Total {len(all_speakers)} unique speakers found.")

            # ----------------------
            # 第 2 步：遍历每个子目录，混合当前目录 2 个说话人 + 0~3 个额外说话人
            # ----------------------
            # list all the subdir in root dir
            sub_dirs = [name for name in os.listdir(now_root_dir) if os.path.isdir(os.path.join(now_root_dir, name))]
            progress_bar = tqdm(total=len(sub_dirs), desc="Mixing", ncols=100)
            spk_count = {2: 0, 3: 0, 4: 0, 5: 0}
            for subdir in sub_dirs:
                progress_bar.update(1)

                spk_ids = subdir.split('_')
                try:
                    spk1_id, spk2_id = spk_ids
                except:
                    raise ValueError(f"Directory name '{subdir}' is not in the correct format.")
                cur_root = os.path.join(now_root_dir, subdir)
                spk1_path = os.path.join(cur_root, 'spk1_reverb.wav')
                spk2_path = os.path.join(cur_root, 'spk2_reverb.wav')


                # -------------------
                # 1) 读取本目录的 2 个说话人音频
                # -------------------
                try:
                    spk1_data, sr = read_wav(spk1_path)
                    spk2_data, _ = read_wav(spk2_path)
                
                    # -------------------
                    # 2) 额外随机选取 0~3 个其他说话人
                    # -------------------
    
                    all_wavs = [spk1_data, spk2_data]
                    num_extra_speakers = random.randint(0, 1)
                    if num_extra_speakers > 0:
                    
                        # 在所有说话人 ID 里排除当前这两个
                        candidate_spk_ids = [sid for sid in all_speakers if sid not in (spk1_id, spk2_id)]
                        extra_spk_ids = random.sample(candidate_spk_ids, num_extra_speakers)
    
                        # -------------------
                        # 3) 收集所有要混合的音频 (含本目录2人 + 额外0~3人)
                        #    先不做增益处理，先把它们都读进来
                        # -------------------
                        # 主说话人设为 spk1_data，后面增益计算以它为参考
                        for spk_id in extra_spk_ids:
                            wav_path = random.choice(speaker_dict[spk_id])
                            wav_data, _ = read_wav(wav_path)
                            all_wavs.append(wav_data)
                    else:
                        extra_spk_ids = []
                    # -------------------
                    # 4) 截断到最短时长
                    # -------------------
                    min_len = min(len(wav) for wav in all_wavs)
                    truncated_wavs = [wav[:min_len] for wav in all_wavs]
    
                    # -------------------
                    # 5) 做增益并叠加
                    #    - 第一个元素 truncated_wavs[0] 视为主说话人，不做增益(或你也可做固定处理)
                    #    - 其余说话人以随机 SIR 相对主说话人做增益
                    # -------------------
                    mix = np.copy(truncated_wavs[0])  # 主说话人
                    main_spk_data = truncated_wavs[0]
                    main_spk_energy = compute_mch_rms_dB(main_spk_data)
                    spks_data_adj = []
                    spks_data_adj.append(main_spk_data)
                    # 对 spk2 + 额外说话人做增益
                    for idx in range(1, len(truncated_wavs)):
                        sir_db = random.uniform(-6, 6)  # 干扰比范围
                        now_spk_data_adj = apply_gain(main_spk_energy, truncated_wavs[idx], sir_db)
                        spks_data_adj.append(now_spk_data_adj)
                        mix += now_spk_data_adj
    
                    # -------------------
                    # 6) 如果有 noise1_reverb.wav，则加噪声
                    # -------------------
                    noise_path = os.path.join(cur_root, 'noise1_reverb.wav')
    
                    noise_data, _ = read_wav(noise_path)
                    # 截断噪声到最短时长(与 mix 对齐)
                    noise_data = noise_data[:min_len]
                    # 以主说话人做参考，随机 SNR
                    snr_db = random.uniform(10, 20)
                    noise_data_adj = apply_gain(main_spk_energy, noise_data, snr_db)
                    mix += noise_data_adj
    
                    # -------------------
                    # 7) 写出结果到新的目录
                    # -------------------
                    all_spk_ids = [spk1_id, spk2_id] + extra_spk_ids
                    out_name = '_'.join(all_spk_ids)
                    out_dir = os.path.join(now_out_root_dir, out_name)
                    os.makedirs(out_dir, exist_ok=True)
    
                    # 8. 保存所有说话人音频
                    for i, spk_id in enumerate(all_spk_ids):
                        save_path = os.path.join(out_dir, f"spk{i+1}.wav")
                        write_wav(save_path, spks_data_adj[i], sr)
    
                    # 9. 保存噪声和混合
                    if noise_data is not None:
                        write_wav(os.path.join(out_dir, "noise.wav"), noise_data, sr)
                    write_wav(os.path.join(out_dir, "mix.wav"), mix, sr)
                    # 记录这是几个说话人混合的结果
                    spk_count[len(all_spk_ids)] += 1
                except:
                    continue
            progress_bar.close()
            print(spk_count)
            print("Done!")
        # break