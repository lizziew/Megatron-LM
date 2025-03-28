# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import List, Sequence

import torch

from megatron.core import parallel_state
from megatron.core.utils import divide, is_torch_min_version

if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base


def split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Args:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def split_tensor_into_1d_equal_chunks(tensor, new_buffer=False):
    """Break a tensor into equal 1D chunks across tensor parallel ranks.

    Returns a Tensor or View with this rank's portion of the data.

    Args:
        tensor: The tensor to split

    Keyword Args:
        new_buffer (bool): If True, returns a new Tensor.
                           If False, returns a view into the existing Tensor.
                           Default is False

    """
    partition_size = torch.numel(tensor) // parallel_state.get_tensor_model_parallel_world_size()
    start_index = partition_size * parallel_state.get_tensor_model_parallel_rank()
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(
            partition_size,
            dtype=tensor.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor):
    """Opposite of split_tensor_into_1d_equal_chunks. Gather values from tensor
    model parallel ranks.

    Returns a new Tensor with the gathered data.

    Args:
        tensor: A Tensor or view of this rank's portion of the data.
    """
    numel_gathered = torch.numel(tensor) * parallel_state.get_tensor_model_parallel_world_size()
    gathered = torch.empty(
        numel_gathered, dtype=tensor.dtype, device=torch.cuda.current_device(), requires_grad=False
    )
    dist_all_gather_func(gathered, tensor, group=parallel_state.get_tensor_model_parallel_group())
    return gathered


class VocabUtility:
    """Split the vocabulary into `world_size` chunks and return the first
    and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [fist, last)

    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        """Vocab range from per partition vocab size."""
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(
        global_vocab_size: int, rank: int, world_size: int
    ) -> Sequence[int]:
        """Vocab range from global vocab size."""
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )


def get_embedding_norms_for_tensor_parallel(
    model,
    token_ids,
    iteration,
    get_min_median_max=True
):
    """Calculate the embedding norms for specified token IDs.
    
    Args:
        model: The model containing the embeddings
        token_ids: List of token IDs to get norms for
        iteration: Current iteration (for logging)
        get_min_median_max: Whether to calculate min/median/max stats
        
    Returns:
        Dictionary mapping token IDs to their embedding norms and
        stats dictionary with min/median/max/avg if requested
    """
    import torch
    import numpy as np
    from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
    from megatron.core.parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    
    if hasattr(model, 'embedding') and hasattr(model.embedding, 'word_embeddings'):
        embedding_layer = model.embedding.word_embeddings
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'embedding'):
        embedding_layer = model.language_model.embedding.word_embeddings
    else:
        raise ValueError("Could not find embedding layer in model")
    
    tokens_tensor = torch.tensor(token_ids, dtype=torch.long, device=torch.cuda.current_device())
    
    with torch.no_grad():
        embeddings = embedding_layer(tokens_tensor)  # [num_tokens, hidden_size]
        
        norms = torch.norm(embeddings, dim=1)  # [num_tokens]
        
    token_norms = {int(token_ids[i]): float(norms[i]) for i in range(len(token_ids))}
    
    stats = {}
    if get_min_median_max and len(norms) > 0:
        norms_np = norms.detach().cpu().numpy()
        stats = {
            'min': float(np.min(norms_np)),
            'max': float(np.max(norms_np)),
            'median': float(np.median(norms_np)),
            'avg': float(np.mean(norms_np))
        }
    
    return token_norms, stats
