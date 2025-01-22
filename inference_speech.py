import yaml
import os
import look2hear.models
import argparse
import torch
import torchaudio


# audio path
parser = argparse.ArgumentParser()
parser.add_argument("--audio_path", default="test/mix.wav", help="Path to audio file")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load model

model = look2hear.models.TIGER.from_pretrained("JusperLee/TIGER-speech", cache_dir="cache")
model.to(device)
model.eval()

audio = torchaudio.load(parser.parse_args().audio_path)[0].to(device)
with torch.no_grad():
    all_target_dialog, all_target_effect, all_target_music = model(audio[None])

torchaudio.save(
    f"test/dialog.wav", all_target_dialog[0].unsqueeze(0).cpu(), 16000
)
torchaudio.save(
    f"test/effect.wav", all_target_effect[0].unsqueeze(0).cpu(), 16000
)
torchaudio.save(
    f"test/music.wav", all_target_music[0].unsqueeze(0).cpu(), 16000
)