import argparse
import json
import os
import soundfile as sf
from tqdm import tqdm
from rich import print


def preprocess_one_dir(in_data_dir, out_dir, data_type, max_speakers=5):
    """Create .json file for one condition with variable number of speakers."""
    # Dictionary to store file information for mix, noise, and each possible speaker
    file_infos = {'mix': [], 'noise': []}
    for i in range(1, max_speakers + 1):
        file_infos[f's{i}'] = []
    
    in_dir = os.path.abspath(os.path.join(in_data_dir, data_type))
    print(f"Process {data_type} set...")
    
    for root, dirs, files in os.walk(in_dir):
        # Find all mix.wav files
        mix_files = [f for f in files if f.endswith(".wav") and f.startswith("mix")]
        
        for mix_file in mix_files:
            # import pdb; pdb.set_trace()

            mix_path = os.path.join(root, mix_file)
            try:
                audio, _ = sf.read(mix_path)
                file_infos['mix'].append((mix_path, len(audio)))
                
                
                # Process speaker files
                for i in range(1, max_speakers + 1):
                    spk_path = os.path.join(root, f"spk{i}.wav")
                    if os.path.exists(spk_path):
                        audio, _ = sf.read(spk_path)
                        file_infos[f's{i}'].append((spk_path, len(audio)))
                    else:
                        file_infos[f's{i}'].append((None, 0))
            
            except Exception as e:
                print(f"Error processing {mix_path}: {e}")
                continue
            
            print(f"Process num: {len(file_infos['mix'])}", end="\r")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(os.path.join(out_dir, data_type)):
        os.makedirs(os.path.join(out_dir, data_type))
    
    # Save information for mix, noise, and each speaker
    for key in file_infos:
        if file_infos[key]:  # Only save if there's data
            with open(os.path.join(out_dir, data_type, f"{key}.json"), "w") as f:
                json.dump(file_infos[key], f, indent=4)
    
    print(f"\nProcessed {len(file_infos['mix'])} files in {data_type} set")
    print(f"Speaker distribution:")
    for i in range(1, max_speakers + 1):
        valid_count = sum(1 for item in file_infos[f's{i}'] if item[0] is not None)
        print(f"  s{i}: {valid_count}/{len(file_infos[f's{i}'])}")


def preprocess_audio_data(inp_args):
    """Create .json files for all conditions."""
    for data_type in ["train", "val", "test"]:
        preprocess_one_dir(
            inp_args.in_dir, 
            inp_args.out_dir, 
            data_type,
            inp_args.max_speakers
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Audio data preprocessing")
    parser.add_argument(
        "--in_dir",
        type=str,
        default="/gpfs-flash/hulab/suyuchang/EchoSet3",
        help="Directory path of audio including train, val and test",
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="/gpfs-flash/hulab/suyuchang/TIGER/DataPreProcess/unknown_speaker/EchoSet3", 
        help="Directory path to put output files"
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=3,
        help="Maximum number of speakers to look for"
    )
    args = parser.parse_args()
    print(args)
    preprocess_audio_data(args)