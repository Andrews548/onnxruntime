// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "expand_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void ExpandKernel2D(
    const size_t N,
    const T* input_data,
    T* output_data,
    const fast_divmod fdm_output_stride0, 
    const int input_view_stride0,
    const int input_view_stride1) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  int dim0, dim1;
  fdm_output_stride0.divmod(id, dim0, dim1);
  output_data[id] = input_data[dim0 * input_view_stride0 + dim1 * input_view_stride1];
}

template <typename T>
__global__ void ExpandKernel(
  const size_t rank,
  const size_t N,
  const size_t N_input,
  const void* input_data,
  void* output_data,
  const fast_divmod* fdm_output_strides,
  const int64_t* input_view_strides) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  int dim, r = id, input_index = 0;
  for (int i = 0; i < rank; ++i) {
    fdm_output_strides[i].divmod(r, dim, r);
    input_index += dim * input_view_strides[i];
  }
  output_data[id] = input_data[input_index];
}

Status ExpandByFill(const size_t element_size, const int N, const void* input_data, void* output_data)
{
#define CASE(TYPE)                                                                                          \
  case sizeof(TYPE):                                                                                        \
    cuda::Fill(reinterpret_cast<TYPE*>(output_data), *(reinterpret_cast<const TYPE*>(input_data)), N);  \
    break

  switch (element_size) {
    CASE(int8_t);
    CASE(int16_t);
    CASE(int32_t);
    CASE(int64_t);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Expand operator");
  }
  return Status::OK();
}

Status Expand2D(
  const size_t element_size,    
  const size_t N,
  const T* input_data,
  T* output_data,
  const fast_divmod fdm_output_stride0, 
  const int input_view_stride0,
  const int input_view_stride1) {
#define CASE(TYPE)                                                                                          \
    case sizeof(TYPE):                                                                                        \
      ExpandKernel2D<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>( \
        element_size, N, reinterpret_cast<TYPE*>(output_data), \
                    *(reinterpret_cast<const TYPE*>(input_data)), fdm_output_stride0, input_view_stride0, input_view_stride1);  \
      break

  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  switch (element_size) {
    CASE(int8_t);
    CASE(int16_t);
    CASE(int32_t);
    CASE(int64_t);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Expand operator");
  }
  return Status::OK();
}

Status ExpandImpl(
  const size_t element_size,
  const int N_output,
  const int N_input,
  const void* input_data,
  void* output_data,
  CudaAsyncBuffer<fast_divmod>& fdm_output_strides, 
  CudaAsyncBuffer<int64_t>& input_view_strides)
{
  int rank = static_cast<int>(fdm_output_strides.Count());
  if (rank == 1) {
    if (N_input == N_output) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_data, input_data, N * element_size, cudaMemcpyDeviceToDevice, 0);
    }
    else { // N_input == 1
      return ExpandByFill(element_size, N_output, input_data, output_data);
    }
  }
  else if (rank == 2) {
    return Expand2D(element_size, N_output, input_data, output_data,
      fdm_output_dims.CpuSpan()[0], input_view_strides.CpuSpan[0], input_view_strides.CpuSpan()[1]);
  }

  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  fdm_output_strides.CopyToGpu();
  input_view_strides.CopyToGpu();
  #define CASE(TYPE)                                                                                          \
    case sizeof(TYPE):                                                                                        \
      ExpandKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>( \
        rank, N, reinterpret_cast<TYPE*>(output_data), *(reinterpret_cast<const TYPE*>(input_data)), \
        fdm_output_strides.GpuPtr(), input_view_strides.GpuPtr());  \
      break

  switch (element_size) {
    CASE(int8_t);
    CASE(int16_t);
    CASE(int32_t);
    CASE(int64_t);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Expand operator");
  }
  return Status::OK();
}


}  // namespace cuda
}  // namespace onnxruntime
