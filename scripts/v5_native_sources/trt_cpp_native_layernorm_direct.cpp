/* Copyright 2026 Hiruynk. See the repository LICENSE. */
// Qualification-only TensorRT 11.2 binding for Native layer_norm(out).
// TensorRT owns all input/output buffers; no pointer survives enqueue().
#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/from_blob.h>
#include <ATen/native/layer_norm.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cstdio>
#include <exception>
#include <new>
#include <optional>

static_assert(NV_TENSORRT_MAJOR == 11 && NV_TENSORRT_MINOR == 2
              && NV_TENSORRT_PATCH == 1 && NV_TENSORRT_BUILD == 2,
              "This diagnostic requires pinned TensorRT 11.2.1.2 headers");

namespace {
using namespace nvinfer1;
constexpr char kName[] = "cpp_native_layernorm_direct";
constexpr char kVersion[] = "1";
constexpr char kNamespace[] = "aniflive_research";
uintptr_t observed_workspace = 0, observed_output = 0;
uintptr_t observed_x = 0, observed_weight = 0, observed_bias = 0;

bool same_dims(Dims const& a, Dims const& b) noexcept {
    if (a.nbDims != b.nbDims) return false;
    for (int i = 0; i < a.nbDims; ++i) if (a.d[i] != b.d[i]) return false;
    return true;
}

bool valid_shapes(PluginTensorDesc const* in, int32_t ni,
                  PluginTensorDesc const* out, int32_t no) noexcept {
    if (!in || !out || ni != 3 || no != 1) return false;
    auto const& x=in[0].dims; auto const& w=in[1].dims; auto const& b=in[2].dims;
    return x.nbDims==3 && x.d[0]==1 && x.d[1]==1 && x.d[2]==512 && w.nbDims==1 && w.d[0]==512
        && b.nbDims==1 && b.d[0]==512 && same_dims(x,out[0].dims);
}

class NativeAttention final : public IPluginV3, public IPluginV3OneCore,
                              public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        if (type == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new (std::nothrow) NativeAttention(); }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return kNamespace; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t getOutputDataTypes(DataType* outputs, int32_t no,
                               DataType const* inputs, int32_t ni) const noexcept override {
        if (!outputs || !inputs || ni != 3 || no != 1 || inputs[0] != DataType::kHALF) return 1;
        outputs[0] = DataType::kHALF;
        return 0;
    }
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t ni, DimsExprs const*, int32_t,
                            DimsExprs* outputs, int32_t no, IExprBuilder&) noexcept override {
        if (!inputs || !outputs || ni != 3 || no != 1 || inputs[0].nbDims != 3) return 1;
        outputs[0] = inputs[0];
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io,
                                    int32_t ni, int32_t no) noexcept override {
        if (!io || ni != 3 || no != 1 || pos < 0 || pos >= 4) return false;
        return io[pos].desc.format == TensorFormat::kLINEAR && io[pos].desc.type == DataType::kHALF;
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t ni,
                            DynamicPluginTensorDesc const*, int32_t no) noexcept override {
        return ni == 3 && no == 1 ? 0 : 1;
    }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t,
                            DynamicPluginTensorDesc const*, int32_t) const noexcept override { return 2*sizeof(float); }
    int32_t onShapeChange(PluginTensorDesc const* in, int32_t ni,
                          PluginTensorDesc const* out, int32_t no) noexcept override {
        return valid_shapes(in, ni, out, no) ? 0 : 1;
    }
    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const* out,
                    void const* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override {
        try {
            if (!valid_shapes(in, 3, out, 1) || !inputs || !outputs) return 1;
            for (int i = 0; i < 3; ++i) if (!inputs[i]) return 1;
            if (!outputs[0] || !workspace) return 1;
            observed_workspace=reinterpret_cast<uintptr_t>(workspace);
            observed_output=reinterpret_cast<uintptr_t>(outputs[0]);
            observed_x=reinterpret_cast<uintptr_t>(inputs[0]);
            observed_weight=reinterpret_cast<uintptr_t>(inputs[1]);
            observed_bias=reinterpret_cast<uintptr_t>(inputs[2]);
            int device = -1;
            if (cudaGetDevice(&device) != cudaSuccess) return 1;
            c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, device));
            c10::InferenceMode inference;
            auto options=at::TensorOptions().dtype(at::kHalf).device(at::kCUDA,device);
            auto x=at::from_blob(const_cast<void*>(inputs[0]),{1,1,512},options);
            auto weight=at::from_blob(const_cast<void*>(inputs[1]),{512},options);
            auto bias=at::from_blob(const_cast<void*>(inputs[2]),{512},options);
            auto target=at::from_blob(outputs[0],{1,1,512},options);
            auto stats_options=at::TensorOptions().dtype(at::kFloat).device(at::kCUDA,device);
            auto mean=at::from_blob(workspace,{1,1,1},stats_options);
            auto rstd=at::from_blob(static_cast<char*>(workspace)+sizeof(float),{1,1,1},stats_options);
            at::native::LayerNormKernel(at::kCUDA,x,weight,bias,1,512,1e-5,&target,&mean,&rstd);
            return 0;
        } catch (std::exception const& error) {
            std::fprintf(stderr, "cpp_native_layernorm_direct enqueue failed: %s\n", error.what());
            return 1;
        } catch (...) {
            std::fprintf(stderr, "cpp_native_layernorm_direct enqueue failed with unknown exception\n");
            return 1;
        }
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &fields_; }
private:
    PluginFieldCollection fields_{0, nullptr};
};

class Creator final : public IPluginCreatorV3One {
public:
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return kNamespace; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fields, TensorRTPhase) noexcept override {
        if (fields && fields->nbFields != 0) return nullptr;
        return new (std::nothrow) NativeAttention();
    }
private:
    PluginFieldCollection fields_{0, nullptr};
};
} // namespace

extern "C" int register_cpp_native_layernorm_direct() noexcept {
    static Creator creator;
    return getPluginRegistry()->registerCreator(creator, kNamespace) ? 0 : 1;
}
extern "C" uintptr_t qualification_layernorm_workspace() noexcept { return observed_workspace; }
extern "C" uintptr_t qualification_layernorm_output() noexcept { return observed_output; }
extern "C" uintptr_t qualification_layernorm_input() noexcept { return observed_x; }
extern "C" uintptr_t qualification_layernorm_weight() noexcept { return observed_weight; }
extern "C" uintptr_t qualification_layernorm_bias() noexcept { return observed_bias; }
