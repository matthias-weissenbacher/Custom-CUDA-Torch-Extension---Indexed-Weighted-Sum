#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <c10/cuda/CUDAGuard.h>



namespace {

    
template <typename scalar_t>
__global__ void trisum_cuda_forward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t>  outputs,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> weights,
    const torch::PackedTensorAccessor<int32_t,2,torch::RestrictPtrTraits,size_t> idxs) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int state_size = input.size(1);
  if (c < state_size){
      if (n < input.size(0)){
          for (int i = 3*c*state_size; i < 3*(c+1)*state_size; i += 1){
             outputs[n][c] += (input[n][idxs[0][i]]-input[n][idxs[1][i]])*weights[i];
      }
      }
  }
}
    


template <typename scalar_t>
__global__ void trisum_cuda_backward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_inputs,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> d_weights,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_out,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> weights,
    const torch::PackedTensorAccessor<int32_t,2,torch::RestrictPtrTraits,size_t> idxs,
    const torch::PackedTensorAccessor<int32_t,2,torch::RestrictPtrTraits,size_t> idxs_bw) {
  //batch index
  const int n = blockIdx.y;
  // column index
  
   
  const int c = blockIdx.x * blockDim.x + threadIdx.x; 
  //const int j = blockIdx.y * blockDim.y + threadIdx.y; 
  
  const int state_size = input.size(1);
  const int batch_size = input.size(0);
 // const int J = 3*c*state_size + j;
  
  if (c < state_size && n < batch_size){
        for (int i = 3*c*state_size; i < 3*(c+1)*state_size; i += 1){      
            d_inputs[n][c] += grad_out[n][idxs_bw[0][i]]*weights[idxs_bw[2][i]]-grad_out[n][idxs_bw[1][i]]*weights[idxs_bw[3][i]];     
            atomicAdd(&d_weights[i], (input[n][idxs[0][i]]-input[n][idxs[1][i]])*grad_out[n][c]);
      }

  }
  
}//end backward kernel
    

    
} // namespace

std::vector<torch::Tensor> trisum_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor idxs,
    torch::Tensor idxs_bw) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const auto batch_size = input.size(0);
  const auto state_size = input.size(1);

  auto outputs =  torch::zeros_like(input);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "trisum_forward_cuda", ([&] {
    trisum_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        outputs.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        weights.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        idxs.packed_accessor<int32_t,2,torch::RestrictPtrTraits,size_t>());
  }));

  return {outputs, input, idxs, idxs_bw};
}

std::vector<torch::Tensor> trisum_cuda_backward(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor idxs,
    torch::Tensor idxs_bw,
    torch::Tensor weights) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto d_inputs = torch::zeros_like(input);
  auto d_weights = torch::zeros_like(weights);

  const auto batch_size = input.size(0);
  const auto state_size = input.size(1);

  //const dim3 threads(1024,1024);
  const int threads = 1024;
  //const int th2 = thr / 2;
  //const dim3 blocks3D((state_size + threads - 1) / threads,  (3*state_size + threads - 1) / threads , batch_size);
  const dim3 blocks( (state_size + threads - 1) / threads, batch_size);
    
    
  //const dim3 threads2D(1024,1024);
  const dim3 blocks2( (state_size + threads - 1) / threads, batch_size);   


  AT_DISPATCH_FLOATING_TYPES(input.type(), "trisum_backward_cuda", ([&] {
    trisum_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_inputs.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_weights.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        grad_out.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        weights.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        idxs.packed_accessor<int32_t,2,torch::RestrictPtrTraits,size_t>(),
        idxs_bw.packed_accessor<int32_t,2,torch::RestrictPtrTraits,size_t>());
  }));

  cudaDeviceSynchronize();

  return {d_inputs, d_weights};
}

