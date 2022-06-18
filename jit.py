from torch.utils.cpp_extension import load
trisum_cuda = load(
    'trisum_cuda', ['trisum_cuda.cpp', 'trisum_cuda_kernel.cu'], verbose=True)
help(trisum_cuda)
