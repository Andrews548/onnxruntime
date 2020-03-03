// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/providers/armnn/nn/conv.h"
#include "core/providers/armnn/armnn_execution_provider.h"
#include "core/providers/armnn/armnn_fwd.h"


#include "armnn/ArmNN.hpp"

#include <thread>
#include <mutex>

namespace onnxruntime {
namespace armnn_ep{

template <typename T>
class FusedConv final : public armnn_ep::Conv<T> {
 public:
  explicit FusedConv(const OpKernelInfo& info) : armnn_ep::Conv<T>(info) {
    ORT_ENFORCE(info.GetAttr<std::string>("activation", &(this->activation_type)).IsOK());
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedConv,
    kMSDomain,
    1,
    float,
    kArmnnExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConv<float>);

}  // namespace armnn_ep
}  // namespace onnxruntime