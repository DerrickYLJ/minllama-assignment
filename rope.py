from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def interleave_tensors(query_real: torch.Tensor, query_imag: torch.Tensor) -> torch.Tensor:
    combined = torch.stack((query_real, query_imag), dim=-1)
    interleaved = combined.flatten(start_dim=-2)
    return interleaved

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bs, seqlen, n_local_heads, d = query.shape
    device = query.device

    # Split query and key into real and imaginary parts
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # Step 1: Calculate the rotary positional encodings
    half_d = d // 2  # Adjust for half the dimension
    freqs = torch.arange(0, half_d, device=device).float()
    freqs = theta ** (-freqs / half_d)
    
    positions = torch.arange(seqlen, device=device).float()
    angles = torch.einsum('n,d->nd', positions, freqs)

    # Compute the cosine and sine values
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    # Step 2: Reshape cos_vals and sin_vals using the helper function
    cos_vals = reshape_for_broadcast(cos_vals, query_real)  # Adjust shape to match real/imag parts
    sin_vals = reshape_for_broadcast(sin_vals, query_real)  # Adjust shape to match real/imag parts

    # Apply RoPE: Combine cos and sin with real and imaginary parts
    query_out = query_real * cos_vals - query_imag * sin_vals
    query_out_imag = query_real * sin_vals + query_imag * cos_vals
    query_out = interleave_tensors(query_out, query_out_imag)

    key_out = key_real * cos_vals - key_imag * sin_vals
    key_out_imag = key_real * sin_vals + key_imag * cos_vals
    key_out = interleave_tensors(key_out, key_out_imag)

    # Return the modified query and key tensors with rotary embeddings
    return query_out, key_out
