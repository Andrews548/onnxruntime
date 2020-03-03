// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/pool.h"
#include "core/providers/armnn/armnn_execution_provider.h"

#include "armnn/ArmNN.hpp"

#include <thread>
#include <mutex>

namespace onnxruntime {
namespace armnn_ep {

typedef struct {
  std::shared_ptr<armnn::NetworkId> networkIdentifier;
} ARMNNPool;

typedef std::map<OpKernel*, ARMNNPool>::iterator PoolLayersIterator;

template <typename T, typename PoolType>
class Pool final : public onnxruntime::Pool<T, PoolType> {
 public:
  explicit Pool(const OpKernelInfo& info) : onnxruntime::Pool<T, PoolType>(info) {
    provider_ = (const_cast<ARMNNExecutionProvider*>(
        dynamic_cast<const ARMNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~Pool() {
    poolLayers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static armnn::IRuntimePtr initRuntime(){
    if(Pool::run)
      return std::move(Pool::run);
    armnn::IRuntime::CreationOptions options;
    return std::move(armnn::IRuntime::Create(options));
  }

 private:
  static thread_local std::map<OpKernel*, ARMNNPool> poolLayers;
  ARMNNExecutionProvider* provider_;
  static thread_local armnn::IRuntimePtr run;
};

}  // namespace armnn_ep
}  // namespace onnxruntime