import csv
import torch
import numpy as np
import logging

from torch_mir_eval.separation import bss_eval_sources
import fast_bss_eval
from ..losses import (
    PITLossWrapper,
    pairwise_neg_sisdr,
    pairwise_neg_snr,
    singlesrc_neg_sisdr,
    PairwiseNegSDR,
)

logger = logging.getLogger(__name__)


class MetricsTracker:
    def __init__(self, save_file: str = ""):
        self.all_sdrs = []
        self.all_sdrs_i = []
        self.all_sisnrs = []
        self.all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]
        self.results_csv = open(save_file, "w")
        self.writer = csv.DictWriter(self.results_csv, fieldnames=csv_columns)
        self.writer.writeheader()
        self.pit_sisnr = PITLossWrapper(
            PairwiseNegSDR("sisdr", zero_mean=False), pit_from="pw_mtx"
        )
        self.pit_snr = PITLossWrapper(
            PairwiseNegSDR("snr", zero_mean=False), pit_from="pw_mtx"
        )

    def __call__(self, mix, clean, estimate, key, n_src):
        # import pdb; pdb.set_trace()
        if n_src is not None and n_src != clean.size(0):
            # Calculate energy for each source
            target_energy = torch.sum(clean ** 2, dim=1)  # [n_src]
            est_energy = torch.sum(estimate ** 2, dim=1)  # [n_src]

            # Get indices of n_src sources with highest energy
            _, target_indices = torch.topk(target_energy, k=n_src)  # [n_src]
            _, est_indices = torch.topk(est_energy, k=n_src)        # [n_src]

            # Select top n_src sources
            filtered_clean = clean[target_indices]       # [n_src, time]
            filtered_estimate = estimate[est_indices]    # [n_src, time]

            # Replace original tensors with filtered ones
            clean = filtered_clean
            estimate = filtered_estimate
        print(clean.shape, estimate.shape)
        # sisnr
        sisnr = self.pit_sisnr(estimate.unsqueeze(0), clean.unsqueeze(0))
        mix = torch.stack([mix] * clean.shape[0], dim=0)
        sisnr_baseline = self.pit_sisnr(mix.unsqueeze(0), clean.unsqueeze(0))
        sisnr_i = sisnr - sisnr_baseline

        # sdr
        sdr = -fast_bss_eval.sdr_pit_loss(estimate, clean).mean()
        sdr_baseline = -fast_bss_eval.sdr_pit_loss(mix, clean).mean()
        sdr_i = sdr - sdr_baseline
        # import pdb; pdb.set_trace()
        row = {
            "snt_id": key,
            "sdr": sdr.item(),
            "sdr_i": sdr_i.item(),
            "si-snr": -sisnr.item(),
            "si-snr_i": -sisnr_i.item(),
        }
        self.writer.writerow(row)
        # Metric Accumulation
        self.all_sdrs.append(sdr.item())
        self.all_sdrs_i.append(sdr_i.item())
        self.all_sisnrs.append(-sisnr.item())
        self.all_sisnrs_i.append(-sisnr_i.item())
    
    def update(self, ):
        return {"sdr_i": np.array(self.all_sdrs_i).mean(),
                "si-snr_i": np.array(self.all_sisnrs_i).mean()
                }

    def final(self,):
        row = {
            "snt_id": "avg",
            "sdr": np.array(self.all_sdrs).mean(),
            "sdr_i": np.array(self.all_sdrs_i).mean(),
            "si-snr": np.array(self.all_sisnrs).mean(),
            "si-snr_i": np.array(self.all_sisnrs_i).mean(),
        }
        self.writer.writerow(row)
        row = {
            "snt_id": "std",
            "sdr": np.array(self.all_sdrs).std(),
            "sdr_i": np.array(self.all_sdrs_i).std(),
            "si-snr": np.array(self.all_sisnrs).std(),
            "si-snr_i": np.array(self.all_sisnrs_i).std(),
        }
        self.writer.writerow(row)
        self.results_csv.close()
