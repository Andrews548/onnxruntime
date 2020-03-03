// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include <cmath>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "core/providers/armnn/nn/pool.h"
#include "core/providers/armnn/armnn_common.h"
#include "core/providers/armnn/armnn_fwd.h"

#include "armnn/ArmNN.hpp"

#define PREF_DIM 4

namespace onnxruntime {
namespace armnn_ep {

template <typename T, typename PoolType>
thread_local std::map<OpKernel*, ARMNNPool> Pool<T, PoolType>::poolLayers;

template <typename T, typename PoolType>
thread_local armnn::IRuntimePtr Pool<T, PoolType>::run = Pool<T, PoolType>::initRuntime();

template <typename T, typename PoolType>
Status Pool<T, PoolType>::Compute(OpKernelContext* context) const {

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  std::vector<int64_t> dilations(PoolBase::pool_attrs_.dilations);
  std::vector<int64_t> armnnDilations(2);
  armnnDilations[0] = (dilations.size() == 2) ? dilations[1] : 1;
  armnnDilations[1] = (!dilations.empty()) ? dilations[0] : 1;

  if ((X->Shape().NumDimensions() != PREF_DIM) ||
      (armnnDilations[0] * armnnDilations[1] > 1)) {
    Status s = onnxruntime::Pool<T, PoolType>::Compute(context);
    return s;
  }

  std::vector<int64_t> pads = PoolBase::pool_attrs_.pads;
  std::vector<int64_t> strides = PoolBase::pool_attrs_.strides;
  std::vector<int64_t> kernel_shape = PoolBase::pool_attrs_.kernel_shape;

  if (PoolBase::pool_attrs_.global_pooling) {
    const auto& input_dims = x_shape.GetDims();
    kernel_shape.assign(input_dims.begin() + 2, input_dims.end());
    strides.assign(kernel_shape.size(), 0);
    pads.assign(kernel_shape.size(), 0);
  }

  std::vector<int64_t> output_dims = PoolBase::pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  Tensor* Y = context->Output(0, TensorShape(output_dims));

  const T* x_data = X->template Data<T>();
  T* y_data = Y->template MutableData<T>();

  ARMNNPool* pPool;
  PoolLayersIterator it = Pool::poolLayers.find((OpKernel*)this);
  if (it == Pool::poolLayers.end()) {

    armnn::PoolingAlgorithm pool_type;
    if (PoolBase::op_name_ == "GlobalAveragePool" || PoolBase::op_name_ == "AveragePool"){
      pool_type = armnn::PoolingAlgorithm::Average;
    } else if (PoolBase::op_name_ == "GlobalMaxPool" || PoolBase::op_name_ == "MaxPool"){
      pool_type = armnn::PoolingAlgorithm::Max;
    } else
      return onnxruntime::Pool<T, PoolType>::Compute(context);

    ARMNNPool tpool;

    armnn::NetworkId networkId;
    tpool.networkIdentifier = std::make_shared<armnn::NetworkId>(networkId);

    armnn::INetworkPtr myNetwork = armnn::INetwork::Create();

    std::vector<int64_t> armnnStrides(2);
    armnnStrides[0] = (strides.size() == 2) ? strides[1] : 1;
    armnnStrides[1] = strides[0];

    std::vector<int64_t> armnnKernelShape(2);
    armnnKernelShape[0] = (kernel_shape.size() > 1) ? kernel_shape[1] : 1;
    armnnKernelShape[1] = kernel_shape[0];

    std::vector<int64_t> armnnPads(4);
    if (pads.size() == 2) {
      if (strides.size() == 1) {
        armnnPads[0] = 0;
        armnnPads[1] = 0;
        armnnPads[2] = pads[1];
        armnnPads[3] = pads[0];
      } else {
        armnnPads[0] = pads[1];
        armnnPads[1] = pads[0];
        armnnPads[2] = pads[1];
        armnnPads[3] = pads[0];
      }
    } else {
      armnnPads[0] = pads[1];
      armnnPads[1] = pads[3];
      armnnPads[2] = pads[0];
      armnnPads[3] = pads[2];
    }

    armnn::Pooling2dDescriptor poolDescriptor;
    poolDescriptor.m_PoolType = pool_type;
    poolDescriptor.m_PadLeft = armnnPads[0];
    poolDescriptor.m_PadRight = armnnPads[1];
    poolDescriptor.m_PadTop = armnnPads[2];
    poolDescriptor.m_PadBottom = armnnPads[3];
    poolDescriptor.m_PoolWidth = armnnKernelShape[0];
    poolDescriptor.m_PoolHeight = armnnKernelShape[1];
    poolDescriptor.m_StrideX = armnnStrides[0];
    poolDescriptor.m_StrideY = armnnStrides[1];
    poolDescriptor.m_OutputShapeRounding = PoolBase::pool_attrs_.ceil_mode ? armnn::OutputShapeRounding::Ceiling : armnn::OutputShapeRounding::Floor;
    poolDescriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    if(pool_type == armnn::PoolingAlgorithm::Average && PoolBase::pool_attrs_.count_include_pad)
      poolDescriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;
    poolDescriptor.m_DataLayout = armnn::DataLayout::NCHW;

    armnn::IConnectableLayer *pool_armnn = myNetwork->AddPooling2dLayer(poolDescriptor, "pool_armnn");
    armnn::TensorShape inputShape = ARMNNTensorShape(X->Shape());
    armnn::TensorShape outputShape = ARMNNTensorShape(Y->Shape());

    armnn::IConnectableLayer *InputLayer  = myNetwork->AddInputLayer(0);
    armnn::IConnectableLayer *OutputLayer = myNetwork->AddOutputLayer(0);

    InputLayer->GetOutputSlot(0).Connect(pool_armnn->GetInputSlot(0));
    pool_armnn->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));

    //Set the tensors in the network.
    armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float32);
    InputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);
    pool_armnn->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Optimise ArmNN network
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*myNetwork, {armnn::Compute::CpuAcc}, Pool::run->GetDeviceSpec());

    // Load graph into runtime
    Pool::run->LoadNetwork(*(tpool.networkIdentifier.get()), std::move(optNet));

    std::pair<PoolLayersIterator, bool> ret;
    ret = Pool::poolLayers.insert(std::pair<OpKernel*, ARMNNPool>((OpKernel*)this, tpool));
    pPool = &ret.first->second;

  } else {
    pPool = &it->second;
  }

  armnn::InputTensors inputTensors{{0, armnn::ConstTensor(Pool::run->GetInputTensorInfo(*(pPool->networkIdentifier.get()), 0),
                                                          x_data)}};
  armnn::OutputTensors outputTensors{{0, armnn::Tensor(Pool::run->GetOutputTensorInfo(*(pPool->networkIdentifier.get()), 0),
                                                       y_data)}};

  // Execute network
  Pool::run->EnqueueWorkload(*(pPool->networkIdentifier.get()), inputTensors, outputTensors);

  return Status::OK();
}

#define POOLING_KERNEL(op_name, data_type, pool_type, since_version, end_version)       \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                              \
      op_name,                                                                          \
      kOnnxDomain,                                                                      \
      since_version,                                                                    \
      end_version,                                                                      \
      data_type,                                                                        \
      kArmnnExecutionProvider,                                                            \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Pool<data_type, pool_type>);

POOLING_KERNEL(MaxPool, float, MaxPool<1>, 1, 7)
POOLING_KERNEL(MaxPool, float, MaxPool<8>, 8, 9)
POOLING_KERNEL(MaxPool, float, MaxPool<8>, 10, 10)
POOLING_KERNEL(AveragePool, float, AveragePool, 7, 9)
POOLING_KERNEL(AveragePool, float, AveragePool, 10, 10)
POOLING_KERNEL(GlobalAveragePool, float, AveragePool, 1, 8)
POOLING_KERNEL(GlobalMaxPool, float, MaxPool<1>, 1, 8)

}  // namespace armnn_ep
}  // namespace onnxruntime


//auto_padd, dilation, global?, 