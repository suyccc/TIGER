import torch

def filter_non_empty_source (clean, estimate, n_src):
    # Calculate energy for each source
    target_energy = torch.sum(clean ** 2, dim=1)  # [n_src]
    est_energy = torch.sum(estimate ** 2, dim=1)  # [n_src
    # Get indices of n_src sources with highest energy
    _, target_indices = torch.topk(target_energy, k=n_src)  # [n_src]
    _, est_indices = torch.topk(est_energy, k=n_src)        # [n_src
    # Select top n_src sources
    filtered_clean = clean[target_indices]       # [n_src, time]
    filtered_estimate = estimate[est_indices]    # [n_src, time
    # Replace original tensors with filtered ones
    clean = filtered_clean
    estimate = filtered_estimate
    return clean, estimate