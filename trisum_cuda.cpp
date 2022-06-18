#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> trisum_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor idxs,
    torch::Tensor idxs_bw);

std::vector<torch::Tensor> trisum_cuda_backward(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor idxs,
    torch::Tensor idxs_bw,
    torch::Tensor weights);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> trisum_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor idxs,
    torch::Tensor idxs_bw) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(idxs);
  CHECK_INPUT(idxs_bw);


  return trisum_cuda_forward(input, weights, idxs, idxs_bw);
}

std::vector<torch::Tensor> trisum_backward(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor idxs,
    torch::Tensor idxs_bw,
    torch::Tensor weights) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(idxs);
  CHECK_INPUT(idxs_bw);

  return trisum_cuda_backward(
      grad_out,
      input,
      idxs,
      idxs_bw,
      weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &trisum_forward, "trisum forward (CUDA)");
  m.def("backward", &trisum_backward, "trisum backward (CUDA)");
}
