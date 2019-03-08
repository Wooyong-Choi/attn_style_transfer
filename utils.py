import torch

def sequence_mask(lengths, hops, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .to(lengths.device)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1))
            .repeat(hops, 1, 1)
            .transpose(0, 1))

def frobenius(matrix):
    # matrix: (bs, r, r)
    bs = matrix.size(0)
    ret = torch.sum((matrix ** 2), 1) # (bs, r)
    ret = torch.sum(ret, 1).squeeze() + 1e-10 # #(bs,), support underflow
    ret = ret ** 0.5
    ret = torch.sum(ret) / bs
    return ret