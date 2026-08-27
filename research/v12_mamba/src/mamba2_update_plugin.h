/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Hiruynk
 * SPDX-License-Identifier: Apache-2.0
 *
 * The recurrent update contract follows the Apache-2.0 Mamba-2 generation
 * path published in NVIDIA TensorRT-LLM v0.18.2. This isolated experiment is
 * not part of the AnifLive-TTS production runtime.
 */

#pragma once

#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace aniflive::mamba
{

inline constexpr char kPluginName[] = "AnifLive-TTS-Mamba2Update";
inline constexpr char kPluginVersion[] = "1";
inline constexpr char kPluginNamespace[] = "aniflive_tts";

int32_t launchMamba2Update(void const* x, void const* state, void const* delta, void const* deltaBias, void const* a,
    void const* b, void const* c, void const* d, void const* z, void* y, void* stateOut, int32_t batch,
    int32_t dim, int32_t nHeads, int32_t nGroups, int32_t dState, bool deltaSoftplus,
    cudaStream_t stream) noexcept;

class Mamba2UpdatePlugin final : public nvinfer1::IPluginV3,
                                 public nvinfer1::IPluginV3OneCore,
                                 public nvinfer1::IPluginV3OneBuild,
                                 public nvinfer1::IPluginV3OneRuntime
{
public:
    Mamba2UpdatePlugin(
        int32_t dim, int32_t nHeads, int32_t nGroups, int32_t dState, bool deltaSoftplus) noexcept;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs,
        int32_t nbOutputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    bool validateShapes(nvinfer1::Dims const* inputs, int32_t nbInputs, nvinfer1::Dims const* outputs,
        int32_t nbOutputs) const noexcept;

    int32_t mDim;
    int32_t mNHeads;
    int32_t mNGroups;
    int32_t mDState;
    int32_t mDeltaSoftplus;
    std::vector<nvinfer1::PluginField> mSerializedFields;
    nvinfer1::PluginFieldCollection mSerializedCollection{};
};

class Mamba2UpdatePluginCreator final : public nvinfer1::IPluginCreatorV3One
{
public:
    Mamba2UpdatePluginCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV3* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fields,
        nvinfer1::TensorRTPhase phase) noexcept override;

private:
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection mFieldCollection{};
};

} // namespace aniflive::mamba

#if defined(_WIN32)
#define ANIFLIVE_PLUGIN_EXPORT __declspec(dllexport)
#else
#define ANIFLIVE_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" ANIFLIVE_PLUGIN_EXPORT bool initAnifLiveTTSMambaPlugins() noexcept;
