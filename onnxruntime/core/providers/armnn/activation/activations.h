// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/activation/activations.h"
#include "core/providers/armnn/armnn_execution_provider.h"

#include "armnn/ArmNN.hpp"

#include <thread>
#include <mutex>

namespace onnxruntime {
namespace armnn_ep {

typedef struct
{
  std::shared_ptr<armnn::NetworkId> networkIdentifier;
} ARMNNRelu;

typedef std::map<OpKernel*, ARMNNRelu>::iterator ReluLayersIterator;

template <typename T>
class Relu : public onnxruntime::Relu<T> {
 public:
  explicit Relu(const OpKernelInfo& info) : onnxruntime::Relu<T>(info) {
  	provider_ = (const_cast<ARMNNExecutionProvider*>(
        dynamic_cast<const ARMNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~Relu() {
  	Relu::reluLayers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static armnn::IRuntimePtr initRuntime(){
  	if(Relu::run)
  		return std::move(Relu::run);
    armnn::IRuntime::CreationOptions options;
    return std::move(armnn::IRuntime::Create(options));
	}

 private:
  static thread_local std::map<OpKernel*, ARMNNRelu> reluLayers;
  ARMNNExecutionProvider* provider_;
  static thread_local armnn::IRuntimePtr run;
};

}  // namespace armnn_ep
}  // namespace onnxruntime
