// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

// Information needed to construct ARMNN execution providers.
struct ARMNNExecutionProviderInfo {
  bool create_arena{true};

  explicit ARMNNExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}

  ARMNNExecutionProviderInfo() = default;
};

// Logical device representation.
class ARMNNExecutionProvider : public IExecutionProvider {
 public:
  explicit ARMNNExecutionProvider(const ARMNNExecutionProviderInfo& info);
  virtual ~ARMNNExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const std::vector<const KernelRegistry*>& kernel_registries) const override;

  const void* GetExecutionHandle() const noexcept override {
    // The ARMNN interface does not return anything interesting.
    return nullptr;
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
};

}  // namespace onnxruntime
