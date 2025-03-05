import argparse
import json
import os
import soundfile as sf
from tqdm import tqdm

def preprocess_one_split(in_data_dir, out_dir, split):
    """Create .json file for one condition."""
    file_infos = {"remix": [], "s1": [], "s2": [], "s3": []}
    in_dir = os.path.abspath(os.path.join(in_data_dir, split))
    in_dir_mix = os.path.join(in_dir, 'remix')
    wav_list = os.listdir(in_dir_mix)
    wav_list.sort()
    for wav_file in tqdm(wav_list):
        if not wav_file.endswith(".wav"):
            continue
        wav_path = os.path.join(in_dir_mix, wav_file)
        samples = sf.SoundFile(wav_path)
        file_infos['remix'].append((wav_path,len(samples)))
        for spk in ['s1', 's2', 's3']:
            in_dir_spk = os.path.join(in_dir, spk)
            wav_path = os.path.join(in_dir_spk, wav_file)
            if not os.path.exists(wav_path):
                file_infos[spk].append((None, 0))
            else:
                samples = sf.SoundFile(wav_path)
                file_infos[spk].append((wav_path,len(samples)))
    
    if not os.path.exists(os.path.join(out_dir, split)):
        os.makedirs(os.path.join(out_dir, split))
    for spk in ['remix', 's1', 's2', 's3']:
        with open(os.path.join(out_dir, split, spk + ".json"), "w") as f:
            json.dump(file_infos[spk], f, indent=4)

def preprocess_librimix_audio(inp_args):
    """Create .json files for all conditions."""
    for split in ["train-100", "dev", "test"]:
            preprocess_one_split(
                inp_args.in_dir, inp_args.out_dir, split
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Librimix audio data preprocessing")
    parser.add_argument(
        "--in_dir",
        type=str,
        default='/gpfs-flash/hulab/suyuchang/LibriMix/',
        help="Directory path of audio including tr, cv and tt",
    )
    parser.add_argument(
        "--out_dir", type=str, default='LibriUmix', help="Directory path to put output files"
    )
    args = parser.parse_args()
    print(args)
    preprocess_librimix_audio(args)