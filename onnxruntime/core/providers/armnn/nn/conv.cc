// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif
#include <thread>
#include <mutex>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "core/providers/armnn/nn/conv.h"
#include "core/providers/armnn/armnn_common.h"
#include "core/providers/armnn/armnn_fwd.h"

#include "armnn/ArmNN.hpp"

#define CONV_ARMNN

#define PREF_DIM 4

namespace onnxruntime {
namespace armnn_ep {

template <typename T>
thread_local std::map<OpKernel*, ARMNNConv> Conv<T>::convLayers;

template <typename T>
thread_local armnn::IRuntimePtr Conv<T>::run = Conv<T>::initRuntime();

#ifdef CONV_ARMNN
template <typename T>
Status Conv<T>::Compute(OpKernelContext* context) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const Tensor* X = context->Input<Tensor>(0);
    const Tensor* W = context->Input<Tensor>(1);
    const Tensor* B = num_inputs == 3 ? context->Input<Tensor>(2) : nullptr;

    const int64_t N = X->Shape()[0];
    const int64_t M = W->Shape()[0];

    if (X->Shape().NumDimensions() != PREF_DIM) {
      Status s = onnxruntime::Conv<T>::Compute(context);
      return s;
    }

    ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

    std::vector<int64_t> kernel_shape;
    ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));

    std::vector<int64_t> pads(conv_attrs_.pads);
    if (pads.empty()) {
      pads.resize(kernel_shape.size() * 2, 0);
    }
    std::vector<int64_t> dilations(conv_attrs_.dilations);
    if (dilations.empty()) {
      dilations.resize(kernel_shape.size(), 1);
    }
    std::vector<int64_t> strides(conv_attrs_.strides);
    if (strides.empty()) {
      strides.resize(kernel_shape.size(), 1);
    }

    std::vector<int64_t> Y_dims;
    Y_dims.insert(Y_dims.begin(), {N, M});
    TensorShape input_shape = X->Shape().Slice(2);
    ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &Y_dims));
    Tensor* Y = context->Output(0, TensorShape(Y_dims));

    bool biasEnabled = B != nullptr;

    const T* x_data = X->template Data<T>();
    const T* k_data = W->template Data<T>();

    const T* b_data;
    if (biasEnabled) {
      b_data = B->template Data<T>();
    }

    T* y_data = Y->template MutableData<T>();

    std::vector<int64_t> armnnStrides(2);
    armnnStrides[0] = (strides.size() == 2) ? strides[1] : 1;
    armnnStrides[1] = strides[0];

    std::vector<int64_t> armnnDilations(2);
    armnnDilations[0] = (dilations.size() == 2) ? dilations[1] : 1;
    armnnDilations[1] = dilations[0];

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

    ARMNNConv* pConv;
    ConvLayersIterator it = Conv::convLayers.find((OpKernel*)this);
    if (it == Conv::convLayers.end()) {

      ARMNNConv tconv;

      armnn::NetworkId networkId;
      tconv.networkIdentifier = std::make_shared<armnn::NetworkId>(networkId);
      armnn::INetworkPtr myNetwork = armnn::INetwork::Create();

      armnn::Convolution2dDescriptor convolutionDescriptor;
      convolutionDescriptor.m_PadLeft = armnnPads[0];
      convolutionDescriptor.m_PadRight = armnnPads[1];
      convolutionDescriptor.m_PadTop = armnnPads[2];
      convolutionDescriptor.m_PadBottom = armnnPads[3];
      convolutionDescriptor.m_StrideX = armnnStrides[0];
      convolutionDescriptor.m_StrideY = armnnStrides[1];
      convolutionDescriptor.m_DilationX = armnnDilations[0];
      convolutionDescriptor.m_DilationY = armnnDilations[1];
      convolutionDescriptor.m_BiasEnabled = biasEnabled;
      convolutionDescriptor.m_DataLayout = armnn::DataLayout::NCHW;

      armnn::IConnectableLayer *convolution_armnn;
      armnn::TensorShape inputShape = ARMNNTensorShape(X->Shape());
      armnn::TensorShape weightShape = ARMNNTensorShape(W->Shape());

      if(weightShape[2] == 1 && weightShape[3] == 1) {
        Status s = onnxruntime::Conv<T>::Compute(context);
        return s;
      }

      if(conv_attrs_.group > 1) {

        if(conv_attrs_.group == inputShape[1]) {
          armnn::DepthwiseConvolution2dDescriptor depthwiseDescriptor;
          depthwiseDescriptor.m_PadLeft      = convolutionDescriptor.m_PadLeft;
          depthwiseDescriptor.m_PadRight     = convolutionDescriptor.m_PadRight;
          depthwiseDescriptor.m_PadTop       = convolutionDescriptor.m_PadTop;
          depthwiseDescriptor.m_PadBottom    = convolutionDescriptor.m_PadBottom;
          depthwiseDescriptor.m_StrideX      = convolutionDescriptor.m_StrideX;
          depthwiseDescriptor.m_StrideY      = convolutionDescriptor.m_StrideY;
          depthwiseDescriptor.m_DilationX    = convolutionDescriptor.m_DilationX;
          depthwiseDescriptor.m_DilationY    = convolutionDescriptor.m_DilationY;
          depthwiseDescriptor.m_BiasEnabled  = convolutionDescriptor.m_BiasEnabled;
          depthwiseDescriptor.m_DataLayout   = convolutionDescriptor.m_DataLayout;

          weightShape[1] = weightShape[0];
          weightShape[0] = 1;
          armnn::TensorInfo weightsInfo(weightShape, armnn::DataType::Float32);
          armnn::ConstTensor weights(weightsInfo, k_data);

          if(biasEnabled){
            armnn::TensorInfo biasDesc(ARMNNTensorShape(B->Shape()), armnn::DataType::Float32);
            armnn::ConstTensor bias(biasDesc, b_data);
            convolution_armnn = myNetwork->AddDepthwiseConvolution2dLayer(depthwiseDescriptor,
                                                                          weights,
                                                                          armnn::Optional<armnn::ConstTensor>(bias),
                                                                          "depthwise_convolution_armnn");
          } else {
            convolution_armnn = myNetwork->AddDepthwiseConvolution2dLayer(depthwiseDescriptor,
                                                                          weights,
                                                                          armnn::EmptyOptional(),
                                                                          "depthwise_convolution_armnn");
          }
        } else {
          Status s = onnxruntime::Conv<T>::Compute(context);
          return s;
        }
      } else {
        armnn::TensorInfo weightsInfo(weightShape, armnn::DataType::Float32);
        armnn::ConstTensor weights(weightsInfo, k_data);

        if(biasEnabled){
          armnn::TensorInfo biasDesc(ARMNNTensorShape(B->Shape()), armnn::DataType::Float32);
          armnn::ConstTensor bias(biasDesc, b_data);
          convolution_armnn = myNetwork->AddConvolution2dLayer(convolutionDescriptor,
                                                               weights,
                                                               armnn::Optional<armnn::ConstTensor>(bias),
                                                               "convolution_armnn");
        } else {
          convolution_armnn = myNetwork->AddConvolution2dLayer(convolutionDescriptor,
                                                               weights,
                                                               armnn::EmptyOptional(),
                                                               "convolution_armnn");
        }
      }

      bool armnn_activ_enabled = false;
      armnn::ActivationDescriptor desc;
      desc.m_A = conv_attrs_.alpha;

      if (activation_type == "Relu") {
        desc.m_Function = armnn::ActivationFunction::ReLu;
        armnn_activ_enabled = true;
      } else if (activation_type == "LeakyRelu") {
        desc.m_Function = armnn::ActivationFunction::LeakyReLu;
        armnn_activ_enabled = true;
      } else if (activation_type == "Tanh") {
        desc.m_Function = armnn::ActivationFunction::TanH;
        armnn_activ_enabled = true;
      } else if (activation_type == "Sigmoid") {
        desc.m_Function = armnn::ActivationFunction::Sigmoid;
        armnn_activ_enabled = true;
      } else if (!activation_type.empty()) {
        ORT_NOT_IMPLEMENTED("Not implemented fused activation: ", activation_type);
      }

      armnn::IConnectableLayer* activation = myNetwork->AddActivationLayer(desc, "activation_armnn");

      armnn::IConnectableLayer *InputLayer  = myNetwork->AddInputLayer(0);
      armnn::IConnectableLayer *OutputLayer = myNetwork->AddOutputLayer(0);

      InputLayer->GetOutputSlot(0).Connect(convolution_armnn->GetInputSlot(0));
      if (armnn_activ_enabled) {
        convolution_armnn->GetOutputSlot(0).Connect(activation->GetInputSlot(0));
        activation->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));
      }
      else {
        convolution_armnn->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));
      }

      //Set the tensors in the network.
      armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float32);
      InputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

      armnn::TensorInfo outputTensorInfo(ARMNNTensorShape(Y->Shape()), armnn::DataType::Float32);
      convolution_armnn->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

      if (armnn_activ_enabled) {
        activation->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
      }

      // Optimise ArmNN network
      armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*myNetwork, {armnn::Compute::CpuAcc}, Conv::run->GetDeviceSpec());

      // Load graph into runtime
      Conv::run->LoadNetwork(*(tconv.networkIdentifier.get()), std::move(optNet));

      std::pair<ConvLayersIterator, bool> ret;
      ret = Conv::convLayers.insert(std::pair<OpKernel*, ARMNNConv>((OpKernel*)this, tconv));
      pConv = &ret.first->second;

    } else {
      pConv = &it->second;
    }

    armnn::InputTensors inputTensors{{0, armnn::ConstTensor(Conv::run->GetInputTensorInfo(*(pConv->networkIdentifier.get()), 0),
                                                            x_data)}};
    armnn::OutputTensors outputTensors{{0, armnn::Tensor(Conv::run->GetOutputTensorInfo(*(pConv->networkIdentifier.get()), 0),
                                                         y_data)}};

    // Execute network
   Conv::run->EnqueueWorkload(*(pConv->networkIdentifier.get()), inputTensors, outputTensors);

    return Status::OK();
}
#else
template <typename T>
Status Conv<T>::Compute(OpKernelContext* context) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();

  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = num_inputs == 3 ? context->Input<Tensor>(2) : nullptr;

  LOGS_DEFAULT(VERBOSE) << "X " << X->Shape().ToString().c_str() << std::endl;
  LOGS_DEFAULT(VERBOSE) << "W " << W->Shape().ToString().c_str() << std::endl;
  if (B != nullptr)
    LOGS_DEFAULT(VERBOSE) << "B " << B->Shape().ToString().c_str() << std::endl;

  Status s = onnxruntime::Conv<T>::Compute(context);
  return s;
}
#endif

ONNX_OPERATOR_KERNEL_EX(
    Conv,
    kOnnxDomain,
    1,
    kArmnnExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

}  // namespace armnn
}  // namespace onnxruntime
