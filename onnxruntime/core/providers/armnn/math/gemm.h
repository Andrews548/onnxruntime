// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/math/gemm.h"
#include "core/providers/armnn/armnn_execution_provider.h"

#define CACHE_TRANSPOSED_DATA

namespace onnxruntime {
namespace armnn_ep {

typedef struct {
  std::shared_ptr<armnn::NetworkId> networkIdentifier;
} ARMNNGEMM;

typedef std::map<OpKernel*, ARMNNGEMM>::iterator GEMMLayersIterator;

template <typename T>
class Gemm : public onnxruntime::Gemm<T> {
 public:
  Gemm(const OpKernelInfo& info) : onnxruntime::Gemm<T>(info) {
    int64_t temp;

    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = temp == 0 ? CblasNoTrans : CblasTrans;
    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = temp == 0 ? CblasNoTrans : CblasTrans;

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const auto X = context->Input<Tensor>(0);
    const auto W = context->Input<Tensor>(1);
    const auto B = context->Input<Tensor>(2);

    GemmHelper helper(X->Shape(), trans_A_ != CblasNoTrans, W->Shape(), trans_B_ != CblasNoTrans, B->Shape());

    if (!helper.State().IsOK())
      return helper.State();

    int64_t M = helper.M();
    int64_t N = helper.N();
    auto Y = context->Output(0, TensorShape({M, N}));

    if(trans_A_ == CblasTrans){ // transpose input
      return onnxruntime::Gemm<T>::Compute(context);
    }

    bool FC = ((alpha_ == 1 && beta_ == 1) || (alpha_ == 1 && beta_ == 0));
    if(!FC){
      return onnxruntime::Gemm<T>::Compute(context);
    }

    int64_t K = helper.K();
    LOGS_DEFAULT(VERBOSE) << "Gemm ACL:" << std::endl;
    if (X) LOGS_DEFAULT(VERBOSE) << "X " << X->Shape().ToString().c_str() << std::endl;
    if (W) LOGS_DEFAULT(VERBOSE) << "W " << W->Shape().ToString().c_str() << std::endl;
    if (B) LOGS_DEFAULT(VERBOSE) << "B " << B->Shape().ToString().c_str() << std::endl;
    LOGS_DEFAULT(VERBOSE) << "Y " << Y->Shape().ToString().c_str() << std::endl;
    LOGS_DEFAULT(VERBOSE) << "M " << (int)M << ", N " << (int)N << ", K " << (int)K << std::endl;
    LOGS_DEFAULT(VERBOSE) << "Alfa " << alpha_ << ", Beta " << beta_ << std::endl;
    LOGS_DEFAULT(VERBOSE) << "trans_A_ " << (trans_A_ == CblasTrans) << std::endl;
    LOGS_DEFAULT(VERBOSE) << "trans_B_ " << (trans_B_ == CblasTrans) << std::endl;
    LOGS_DEFAULT(VERBOSE) << std::endl;

    const T* a_data = X->template Data<T>();
    const T* b_data = W->template Data<T>();
    const T* c_data;
    if(B != nullptr && beta_ != 0)
      c_data = B->template Data<T>();
    T* d_data = Y->template MutableData<T>();

    ARMNNGEMM* pGemm;
    GEMMLayersIterator it = Gemm::gemmLayers.find((OpKernel*)this);
    if (it == Gemm::gemmLayers.end()) {
      ARMNNGEMM tgemm;
      
      armnn::NetworkId networkId;
      tgemm.networkIdentifier = std::make_shared<armnn::NetworkId>(networkId);
      armnn::INetworkPtr myNetwork = armnn::INetwork::Create();

      armnn::TensorShape inputShape = ARMNNTensorShape(X->Shape());
      armnn::TensorShape weightShape = ARMNNTensorShape(W->Shape());
      armnn::TensorShape outputShape = ARMNNTensorShape(Y->Shape());

      armnn::FullyConnectedDescriptor fcDescriptor;
      fcDescriptor.m_BiasEnabled = B != nullptr && beta_ != 0;
      fcDescriptor.m_TransposeWeightMatrix = trans_B_ == CblasTrans;

      armnn::IConnectableLayer* fc_armnn;

      armnn::TensorInfo weightsInfo(weightShape, armnn::DataType::Float32);
      armnn::ConstTensor weights(weightsInfo, b_data);

      if(fcDescriptor.m_BiasEnabled){
        armnn::TensorShape biasShape = ARMNNTensorShape(B->Shape());
        if(B->Shape().NumDimensions() == 2){
          if(B->Shape().GetDims()[0] == 1 && B->Shape().GetDims()[1] > 1)
            biasShape = {B->Shape().GetDims()[1]};
        }
        armnn::TensorInfo biasDesc(biasShape, armnn::DataType::Float32);
        armnn::ConstTensor bias(biasDesc, c_data);
        fc_armnn = myNetwork->AddFullyConnectedLayer(fcDescriptor,
                                                     weights,
                                                     armnn::Optional<armnn::ConstTensor>(bias),
                                                     "fc_armnn");
      } else {
        fc_armnn = myNetwork->AddFullyConnectedLayer(fcDescriptor,
                                                     weights,
                                                     armnn::EmptyOptional(),
                                                     "fc_armnn");
      }

      armnn::IConnectableLayer *InputLayer  = myNetwork->AddInputLayer(0);
      armnn::IConnectableLayer *OutputLayer = myNetwork->AddOutputLayer(0);

      InputLayer->GetOutputSlot(0).Connect(fc_armnn->GetInputSlot(0));
      fc_armnn->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));

      //Set the tensors in the network.
      armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float32);
      InputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

      armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);
      fc_armnn->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

      // Optimise ArmNN network
      armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*myNetwork, {armnn::Compute::CpuAcc}, Gemm::run->GetDeviceSpec());

      if(optNet == nullptr){
        return onnxruntime::Gemm<T>::Compute(context);
      }

      // Load graph into runtime
      Gemm::run->LoadNetwork(*(tgemm.networkIdentifier.get()), std::move(optNet));


      std::pair<GEMMLayersIterator, bool> ret;
      ret = Gemm::gemmLayers.insert(std::pair<OpKernel*, ARMNNGEMM>((OpKernel*)this, tgemm));
      pGemm = &ret.first->second;

    } else {
      pGemm = &it->second;
    }

    armnn::InputTensors inputTensors{{0, armnn::ConstTensor(Gemm::run->GetInputTensorInfo(*(pGemm->networkIdentifier.get()), 0),
                                                          a_data)}};
    armnn::OutputTensors outputTensors{{0, armnn::Tensor(Gemm::run->GetOutputTensorInfo(*(pGemm->networkIdentifier.get()), 0),
                                                         d_data)}};

    Gemm::run->EnqueueWorkload(*(pGemm->networkIdentifier.get()), inputTensors, outputTensors);

    return Status::OK();
  }

  ~Gemm() {
    gemmLayers.erase(this);
  }

  static armnn::IRuntimePtr initRuntime(){
    if(Gemm::run)
      return std::move(Gemm::run);
    armnn::IRuntime::CreationOptions options;
    return std::move(armnn::IRuntime::Create(options));
  }

 private:
  static thread_local std::map<OpKernel*, ARMNNGEMM> gemmLayers;
  ARMNNExecutionProvider* provider_;
  static thread_local armnn::IRuntimePtr run;

  CBLAS_TRANSPOSE trans_A_;
  CBLAS_TRANSPOSE trans_B_;
  float alpha_;
  float beta_;
};

template <typename T>
thread_local std::map<OpKernel*, ARMNNGEMM> onnxruntime::armnn_ep::Gemm<T>::gemmLayers;

template <typename T>
thread_local armnn::IRuntimePtr Gemm<T>::run = Gemm<T>::initRuntime();

}  // namespace armnn_ep
}  // namespace onnxruntime
