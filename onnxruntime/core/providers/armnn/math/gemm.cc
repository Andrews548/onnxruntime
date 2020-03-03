// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/armnn/armnn_common.h"
#include "core/providers/armnn/math/gemm.h"
#include "core/providers/armnn/armnn_fwd.h"

namespace onnxruntime {
namespace armnn_ep {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    7,
    9,
    kArmnnExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

}  // namespace armnn_ep
}  // namespace onnxruntime
