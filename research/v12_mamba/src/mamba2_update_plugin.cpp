/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Hiruynk
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mamba2_update_plugin.h"

#include <NvInferRuntimeCommon.h>

#include <algorithm>
#include <cstring>
#include <memory>

namespace aniflive::mamba
{
namespace
{

constexpr int32_t kInputCount = 9;
constexpr int32_t kOutputCount = 2;
constexpr int32_t kSuccess = 0;
constexpr int32_t kFailure = 1;

bool compatibleDimension(int32_t actual, int32_t expected) noexcept
{
    return actual == -1 || actual == expected;
}

bool sameDynamicDimension(int32_t lhs, int32_t rhs) noexcept
{
    return lhs == -1 || rhs == -1 || lhs == rhs;
}

bool parseIntField(nvinfer1::PluginFieldCollection const* fields, char const* name, int32_t& value) noexcept
{
    if (fields == nullptr || fields->fields == nullptr)
    {
        return false;
    }
    for (int32_t index = 0; index < fields->nbFields; ++index)
    {
        nvinfer1::PluginField const& field = fields->fields[index];
        if (field.name != nullptr && field.data != nullptr && std::strcmp(field.name, name) == 0
            && field.type == nvinfer1::PluginFieldType::kINT32 && field.length == 1)
        {
            value = *static_cast<int32_t const*>(field.data);
            return true;
        }
    }
    return false;
}

} // namespace

Mamba2UpdatePlugin::Mamba2UpdatePlugin(
    int32_t dim, int32_t nHeads, int32_t nGroups, int32_t dState, bool deltaSoftplus) noexcept
    : mDim(dim)
    , mNHeads(nHeads)
    , mNGroups(nGroups)
    , mDState(dState)
    , mDeltaSoftplus(deltaSoftplus ? 1 : 0)
{
}

nvinfer1::IPluginCapability* Mamba2UpdatePlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept
{
    if (type == nvinfer1::PluginCapabilityType::kCORE)
    {
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kBUILD)
    {
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kRUNTIME)
    {
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

char const* Mamba2UpdatePlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Mamba2UpdatePlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Mamba2UpdatePlugin::getPluginNamespace() const noexcept
{
    return kPluginNamespace;
}

int32_t Mamba2UpdatePlugin::getNbOutputs() const noexcept
{
    return kOutputCount;
}

nvinfer1::IPluginV3* Mamba2UpdatePlugin::clone() noexcept
{
    return new (std::nothrow) Mamba2UpdatePlugin(mDim, mNHeads, mNGroups, mDState, mDeltaSoftplus != 0);
}

bool Mamba2UpdatePlugin::supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != kInputCount || nbOutputs != kOutputCount || pos < 0
        || pos >= nbInputs + nbOutputs || inOut[pos].desc.format != nvinfer1::TensorFormat::kLINEAR)
    {
        return false;
    }

    switch (pos)
    {
    case 3:
    case 4:
    case 7:
        return inOut[pos].desc.type == nvinfer1::DataType::kFLOAT;
    default:
        return inOut[pos].desc.type == nvinfer1::DataType::kHALF;
    }
}

int32_t Mamba2UpdatePlugin::getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != kInputCount || nbOutputs != kOutputCount)
    {
        return kFailure;
    }
    outputTypes[0] = inputTypes[0];
    outputTypes[1] = inputTypes[1];
    return kSuccess;
}

int32_t Mamba2UpdatePlugin::getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    static_cast<void>(shapeInputs);
    static_cast<void>(nbShapeInputs);
    static_cast<void>(exprBuilder);
    if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount || nbOutputs != kOutputCount)
    {
        return kFailure;
    }
    outputs[0] = inputs[0];
    outputs[1] = inputs[1];
    return kSuccess;
}

bool Mamba2UpdatePlugin::validateShapes(nvinfer1::Dims const* inputs, int32_t nbInputs,
    nvinfer1::Dims const* outputs, int32_t nbOutputs) const noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kInputCount || nbOutputs != kOutputCount
        || mDim <= 0 || mNHeads <= 0 || mNGroups <= 0 || mDState <= 0 || mDState > 128 || mDim % mNHeads != 0
        || mNHeads % mNGroups != 0)
    {
        return false;
    }

    int32_t const headDim = mDim / mNHeads;
    bool valid = inputs[0].nbDims == 2 && inputs[1].nbDims == 4 && inputs[2].nbDims == 2
        && inputs[3].nbDims == 1 && inputs[4].nbDims == 1 && inputs[5].nbDims == 3 && inputs[6].nbDims == 3
        && inputs[7].nbDims == 1 && inputs[8].nbDims == 2 && outputs[0].nbDims == 2 && outputs[1].nbDims == 4;
    if (!valid)
    {
        return false;
    }

    int32_t const batch = inputs[0].d[0];
    valid = compatibleDimension(inputs[0].d[1], mDim) && compatibleDimension(inputs[1].d[1], mNHeads)
        && compatibleDimension(inputs[1].d[2], mDState) && compatibleDimension(inputs[1].d[3], headDim)
        && compatibleDimension(inputs[2].d[1], mNHeads) && compatibleDimension(inputs[3].d[0], mNHeads)
        && compatibleDimension(inputs[4].d[0], mNHeads) && compatibleDimension(inputs[5].d[1], mNGroups)
        && compatibleDimension(inputs[5].d[2], mDState) && compatibleDimension(inputs[6].d[1], mNGroups)
        && compatibleDimension(inputs[6].d[2], mDState) && compatibleDimension(inputs[7].d[0], mNHeads)
        && compatibleDimension(inputs[8].d[1], mDim) && compatibleDimension(outputs[0].d[1], mDim)
        && compatibleDimension(outputs[1].d[1], mNHeads) && compatibleDimension(outputs[1].d[2], mDState)
        && compatibleDimension(outputs[1].d[3], headDim);
    for (int32_t index : {1, 2, 5, 6, 8})
    {
        valid = valid && sameDynamicDimension(batch, inputs[index].d[0]);
    }
    valid = valid && sameDynamicDimension(batch, outputs[0].d[0])
        && sameDynamicDimension(batch, outputs[1].d[0]);
    return valid;
}

int32_t Mamba2UpdatePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    if (inputs == nullptr || outputs == nullptr)
    {
        return kFailure;
    }
    nvinfer1::Dims inputDims[kInputCount];
    nvinfer1::Dims outputDims[kOutputCount];
    for (int32_t index = 0; index < kInputCount; ++index)
    {
        inputDims[index] = inputs[index].desc.dims;
    }
    for (int32_t index = 0; index < kOutputCount; ++index)
    {
        outputDims[index] = outputs[index].desc.dims;
    }
    return validateShapes(inputDims, nbInputs, outputDims, nbOutputs) ? kSuccess : kFailure;
}

size_t Mamba2UpdatePlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    static_cast<void>(inputs);
    static_cast<void>(nbInputs);
    static_cast<void>(outputs);
    static_cast<void>(nbOutputs);
    return 0;
}

int32_t Mamba2UpdatePlugin::onShapeChange(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    if (inputs == nullptr || outputs == nullptr)
    {
        return kFailure;
    }
    nvinfer1::Dims inputDims[kInputCount];
    nvinfer1::Dims outputDims[kOutputCount];
    for (int32_t index = 0; index < kInputCount; ++index)
    {
        inputDims[index] = inputs[index].dims;
    }
    for (int32_t index = 0; index < kOutputCount; ++index)
    {
        outputDims[index] = outputs[index].dims;
    }
    return validateShapes(inputDims, nbInputs, outputDims, nbOutputs) ? kSuccess : kFailure;
}

int32_t Mamba2UpdatePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    static_cast<void>(outputDesc);
    static_cast<void>(workspace);
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr)
    {
        return kFailure;
    }
    int32_t const batch = inputDesc[0].dims.d[0];
    return launchMamba2Update(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7],
        inputs[8], outputs[0], outputs[1], batch, mDim, mNHeads, mNGroups, mDState, mDeltaSoftplus != 0, stream);
}

nvinfer1::IPluginV3* Mamba2UpdatePlugin::attachToContext(nvinfer1::IPluginResourceContext* context) noexcept
{
    static_cast<void>(context);
    return clone();
}

nvinfer1::PluginFieldCollection const* Mamba2UpdatePlugin::getFieldsToSerialize() noexcept
{
    mSerializedFields.clear();
    mSerializedFields.emplace_back("dim", &mDim, nvinfer1::PluginFieldType::kINT32, 1);
    mSerializedFields.emplace_back("nheads", &mNHeads, nvinfer1::PluginFieldType::kINT32, 1);
    mSerializedFields.emplace_back("ngroups", &mNGroups, nvinfer1::PluginFieldType::kINT32, 1);
    mSerializedFields.emplace_back("dstate", &mDState, nvinfer1::PluginFieldType::kINT32, 1);
    mSerializedFields.emplace_back(
        "delta_softplus", &mDeltaSoftplus, nvinfer1::PluginFieldType::kINT32, 1);
    mSerializedCollection.nbFields = static_cast<int32_t>(mSerializedFields.size());
    mSerializedCollection.fields = mSerializedFields.data();
    return &mSerializedCollection;
}

Mamba2UpdatePluginCreator::Mamba2UpdatePluginCreator()
{
    mFields.emplace_back("dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mFields.emplace_back("nheads", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mFields.emplace_back("ngroups", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mFields.emplace_back("dstate", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mFields.emplace_back("delta_softplus", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mFieldCollection.nbFields = static_cast<int32_t>(mFields.size());
    mFieldCollection.fields = mFields.data();
}

char const* Mamba2UpdatePluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Mamba2UpdatePluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Mamba2UpdatePluginCreator::getPluginNamespace() const noexcept
{
    return kPluginNamespace;
}

nvinfer1::PluginFieldCollection const* Mamba2UpdatePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

nvinfer1::IPluginV3* Mamba2UpdatePluginCreator::createPlugin(char const* name,
    nvinfer1::PluginFieldCollection const* fields, nvinfer1::TensorRTPhase phase) noexcept
{
    static_cast<void>(name);
    static_cast<void>(phase);
    int32_t dim = 0;
    int32_t nHeads = 0;
    int32_t nGroups = 0;
    int32_t dState = 0;
    int32_t deltaSoftplus = 0;
    if (!parseIntField(fields, "dim", dim) || !parseIntField(fields, "nheads", nHeads)
        || !parseIntField(fields, "ngroups", nGroups) || !parseIntField(fields, "dstate", dState)
        || !parseIntField(fields, "delta_softplus", deltaSoftplus) || dim <= 0 || nHeads <= 0 || nGroups <= 0
        || dState <= 0 || dState > 128 || dim % nHeads != 0 || nHeads % nGroups != 0)
    {
        return nullptr;
    }
    return new (std::nothrow) Mamba2UpdatePlugin(dim, nHeads, nGroups, dState, deltaSoftplus != 0);
}

} // namespace aniflive::mamba

extern "C" ANIFLIVE_PLUGIN_EXPORT bool initAnifLiveTTSMambaPlugins() noexcept
{
    nvinfer1::IPluginRegistry* registry = ::getPluginRegistry();
    if (registry == nullptr)
    {
        return false;
    }
    if (registry->getCreator(
            aniflive::mamba::kPluginName, aniflive::mamba::kPluginVersion, aniflive::mamba::kPluginNamespace)
        != nullptr)
    {
        return true;
    }
    static aniflive::mamba::Mamba2UpdatePluginCreator creator;
    return registry->registerCreator(creator, aniflive::mamba::kPluginNamespace);
}
