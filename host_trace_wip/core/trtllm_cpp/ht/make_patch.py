"""Writes land/core/trtllm_cpp/src (the patched launcher) from upstream/src (FlashInfer 0.6.18's files, hashes in
src/UPSTREAM.sha256). Every change is one substitution below, each asserted to match exactly once, so the patch
is reviewable here and re-applies to a new FlashInfer by rerunning (a substitution that no longer matches fails).

The changes are types: a value that is symbolic under a host trace (a size, a count, an address) gets an fi_ht
alias (ht/fi_ht.h: the original type in FlashInfer's own build, c10::SymInt under FI_HT_TRACED). The few others
are marked: helper calls where an address is cast or tested, and #ifdef FI_HT_TRACED bodies where the traced build
does something else (the cost model declines, the preload, code it does not compile).
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UP = os.path.join(ROOT, "upstream/src")
OUT = os.environ.get("FI_HT_PATCH_OUT", os.path.join(ROOT, "src"))
FMHA = "include/flashinfer/trtllm/fmha"

files: dict[str, str] = {}


def load(rel: str) -> None:
    with open(os.path.join(UP, rel)) as f:
        files[rel] = f.read()


def sub(rel: str, old: str, new: str, count: int = 1, before: str | None = None) -> None:
    """Replace `old` (exactly `count` times) in the file, or in its text ahead of `before`."""
    s = files[rel]
    cut = s.index(before) if before else len(s)
    head, tail = s[:cut], s[cut:]
    n = head.count(old)
    if n != count:
        sys.exit(f"{rel}: expected {count} match(es) of {old!r}, found {n}")
    files[rel] = head.replace(old, new) + tail


def sub_in(rel: str, start: str, end: str, pattern: str, repl: str, expect: int) -> None:
    """A regex substitution limited to the text between `start` and `end` (a struct's members)."""
    s = files[rel]
    a = s.index(start)
    b = s.index(end, a)
    body, n = re.subn(pattern, repl, s[a:b], flags=re.M)
    if n != expect:
        sys.exit(f"{rel}: expected {expect} substitutions of {pattern!r}, made {n}")
    files[rel] = s[:a] + body + s[b:]


# ---------------------------------------------------------------- fmhaRunnerParams.h
P = f"{FMHA}/fmhaRunnerParams.h"
load(P)
sub(P, '#include "flashinfer/exception.h"\n',
    '#include "flashinfer/exception.h"\n#include <fi_ht.h>\n\n#ifndef FI_HT_TRACED  // the traced build uses the stock enums\n')
sub(P, "#undef MULTI_CTAS_KV_MODE_FUNCTION\n", "#undef MULTI_CTAS_KV_MODE_FUNCTION\n#endif  // FI_HT_TRACED\n")
# every address member of the runner params
sub_in(P, "struct TllmGenFmhaRunnerParams {", "struct TllmGenSelectKernelParams {",
       r"^  ((?:void|uint32_t|int64_t|int32_t|int|float|float2)(?: const)?)\* (\w+)(\{nullptr\})?;",
       r"  fi_ht::ptr<\1> \2\3;", 31)
for name in ("mBatchSize", "mMaxSeqLenQ", "mMaxSeqLenKv", "mSumOfSeqLensQ", "mSumOfSeqLensKv", "mMaxNumPagesPerSeqKv"):
    sub(P, f"  int {name};", f"  fi_ht::sint {name};")
sub(P, "  int64_t lseStrideTokens;\n  int64_t lseStrideHeads;", "  fi_ht::s64 lseStrideTokens;\n  fi_ht::s64 lseStrideHeads;")

# ---------------------------------------------------------------- kernelParams.h
K = f"{FMHA}/kernelParams.h"
load(K)
# the members: the traced build's are proxies of the stock struct's (gen_fields.py)
sub(K, "struct KernelParams {\n", "struct KernelParams\n#ifdef FI_HT_TRACED\n    : fi_ht::KernelParamsFields\n#endif\n{\n#ifndef FI_HT_TRACED\n")
sub(K, "  bool mUsesSharedPagedKvIdx{true};\n", "  bool mUsesSharedPagedKvIdx{true};\n#endif  // FI_HT_TRACED\n")
# TMA shapes and strides
sub(K, "std::vector<uint64_t>", "std::vector<fi_ht::u64>", 13)


def parenthesize_vector_lists(rel: str, head: str, expect: int) -> None:
    """`head{...}` -> `head({...})`: nvcc's front end (cudafe) builds `auto s = std::vector<c10::SymInt>{a, b, c}`
    inside a template with one element (tests/test_nvcc_vector_init.cu); the parenthesized list is the same
    initializer_list constructor in either build."""
    s = files[rel]
    out, i, n = [], 0, 0
    while (j := s.find(head + "{", i)) >= 0:
        k, depth = j + len(head), 0
        while True:
            depth += {"{": 1, "}": -1}.get(s[k], 0)
            k += 1
            if depth == 0:
                break
        out += [s[i:j], head, "(", s[j + len(head) : k], ")"]
        i, n = k, n + 1
    if n != expect:
        sys.exit(f"{rel}: expected {expect} {head}{{...}} lists, found {n}")
    files[rel] = "".join(out) + s[i:]


parenthesize_vector_lists(K, "std::vector<fi_ht::u64>", 10)
parenthesize_vector_lists(K, "std::vector<uint32_t>", 2)
sub(K, "static_cast<uint64_t>(", "static_cast<fi_ht::u64>(", 33)
sub(K, "    int32_t numTokens{options.mSumOfSeqLensQ};", "    fi_ht::s32 numTokens{options.mSumOfSeqLensQ};", 2)
sub(K, "    int32_t numKeysVals{options.mMaxSeqLenKv};", "    fi_ht::s32 numKeysVals{options.mMaxSeqLenKv};")
sub(K, "    int32_t batchSize{options.mBatchSize};", "    fi_ht::s32 batchSize{options.mBatchSize};")
# addresses
sub(K, "  static std::tuple<void const*, void const*, void const*> getDevicePtrs(",
    "  static std::tuple<fi_ht::ptr<void const>, fi_ht::ptr<void const>, fi_ht::ptr<void const>> getDevicePtrs(")
sub(K, "    void const *qPtr{runnerParams.qPtr}, *kPtr{runnerParams.kPtr}, *vPtr{runnerParams.vPtr};",
    "    fi_ht::ptr<void const> qPtr{runnerParams.qPtr}, kPtr{runnerParams.kPtr}, vPtr{runnerParams.vPtr};")
sub(K, "reinterpret_cast<void const*>(reinterpret_cast<char const*>(runnerParams.qkvPtr) +", "fi_ht::byte_offset(runnerParams.qkvPtr,", 2)
sub(K, "reinterpret_cast<void const*>(reinterpret_cast<char const*>(runnerParams.kvPtr) +", "fi_ht::byte_offset(runnerParams.kvPtr,")
sub(K, "const_cast<void*>(", "fi_ht::const_cast_void(", 7)
sub(K, "std::vector<uint32_t> const& tileShapes, void* gmemAddr,", "std::vector<uint32_t> const& tileShapes, fi_ht::ptr<void> gmemAddr,")
sub(K, "    FLASHINFER_CHECK((reinterpret_cast<uint64_t>(gmemAddr) & 0b1111) == 0);", "    FLASHINFER_CHECK(fi_ht::aligned(gmemAddr, 16));")
sub(K, "                                      int32_t maxNumCtasQ, int32_t maxNumCtasKv) {",
    "                                      fi_ht::s32 maxNumCtasQ, fi_ht::s32 maxNumCtasKv) {")
sub(K, "    params.ptrPartialStats = reinterpret_cast<float2*>(options.multiCtasKvScratchPtr);",
    "    params.ptrPartialStats = fi_ht::ptr_cast<float2*>(options.multiCtasKvScratchPtr);")

# ---------------------------------------------------------------- fmhaRunner.cuh
R = f"{FMHA}/fmhaRunner.cuh"
load(R)
# the parentheses keep argument-dependent lookup (Data_type is a global enum) from also finding the stock function
sub(R, "    mKernel =\n        getTllmFmhaKernels(mDtypeQ,", "    mKernel =\n        (getTllmFmhaKernels)(mDtypeQ,")

# ---------------------------------------------------------------- lse.cuh
L = f"{FMHA}/lse.cuh"
load(L)
sub(L, '#include "../../utils.cuh"\n', '#include "../../utils.cuh"\n#include <fi_ht.h>\n')
# its own include guard (the traced build includes the stock file too)
sub(L, "FLASHINFER_TRTLLM_FMHA_LSE_CUH", "FLASHINFER_TRTLLM_FMHA_LSE_CUH_FI_HT", 3)
sub(L, "inline cudaError_t ComputeLSEFromMD(float2* md, float* lse, int num_tokens, int num_heads_q,\n"
       "                                    int64_t lse_stride_tokens, int64_t lse_stride_heads,",
    "inline cudaError_t ComputeLSEFromMD(fi_ht::ptr<float2> md, fi_ht::ptr<float> lse, fi_ht::sint num_tokens,\n"
    "                                    int num_heads_q, fi_ht::s64 lse_stride_tokens, fi_ht::s64 lse_stride_heads,")
sub(L, "  int n = num_tokens * num_heads_q;\n  int num_threads = std::min(1024, UpPowerOfTwo(n));\n  int num_blocks = ceil_div(n, num_threads);",
    "  fi_ht::sint n = num_tokens * num_heads_q;\n  fi_ht::sint num_threads = std::min<fi_ht::sint>(1024, UpPowerOfTwo(n));\n"
    "  fi_ht::sint num_blocks = ceil_div(n, num_threads);")

# ---------------------------------------------------------------- fmhaKernels.cuh
F = f"{FMHA}/fmhaKernels.cuh"
load(F)
sub(F, "namespace flashinfer::trtllm_cubin_loader {\nstd::string getCubin(const std::string& kernelName, const std::string& sha256);\n"
       "}  // namespace flashinfer::trtllm_cubin_loader\nusing flashinfer::trtllm_cubin_loader::getCubin;\n",
    "#ifndef FI_HT_TRACED  // the traced build loads through the stock loader\n"
    "namespace flashinfer::trtllm_cubin_loader {\nstd::string getCubin(const std::string& kernelName, const std::string& sha256);\n"
    "}  // namespace flashinfer::trtllm_cubin_loader\nusing flashinfer::trtllm_cubin_loader::getCubin;\n#endif\n")
for m in ("mMaxNumCtasQ", "mMaxNumCtasKv", "mNumCtasX", "mNumCtasZ", "mClusterDimX"):
    sub(F, f"    int {m};", f"    fi_ht::sint {m};")
# computeCtaAndClusterConfig
sub(F, "    int numCtasPerSeqQ = (params.mMaxSeqLenQ + kernelMeta.mStepQ - 1) / kernelMeta.mStepQ;",
    "    fi_ht::sint numCtasPerSeqQ = (params.mMaxSeqLenQ + kernelMeta.mStepQ - 1) / kernelMeta.mStepQ;")
sub(F, "    int numCtasX = numCtasPerSeqQ;", "    fi_ht::sint numCtasX = numCtasPerSeqQ;")
sub(F, "    int numCtasZ = params.mBatchSize;", "    fi_ht::sint numCtasZ = params.mBatchSize;")
sub(F, "    int numCtasPerSeqKv = 1;", "    fi_ht::sint numCtasPerSeqKv = 1;")
sub(F, "      int maxAttentionWindow{params.mMaxSeqLenKv};", "      fi_ht::sint maxAttentionWindow{params.mMaxSeqLenKv};")
sub(F, "          std::min(params.mMaxSeqLenKv, params.mAttentionWindowSize + kernelMeta.mStepKv - 1);",
    "          std::min<fi_ht::sint>(params.mMaxSeqLenKv, params.mAttentionWindowSize + kernelMeta.mStepKv - 1);")
sub(F, "          maxAttentionWindow = std::min(params.mMaxSeqLenKv, params.mChunkedAttentionSize);",
    "          maxAttentionWindow = std::min<fi_ht::sint>(params.mMaxSeqLenKv, params.mChunkedAttentionSize);")
sub(F, "        maxAttentionWindow = std::min(params.mMaxSeqLenKv, params.mSparseMlaTopK);",
    "        maxAttentionWindow = std::min<fi_ht::sint>(params.mMaxSeqLenKv, params.mSparseMlaTopK);")
sub(F, "      int const maxNumCtasPerSeqKv =\n          (maxAttentionWindow", "      fi_ht::sint const maxNumCtasPerSeqKv =\n          (maxAttentionWindow")
sub(F, "      int tunedMaxNumCtasPerSeqKv = maxNumCtasPerSeqKv;", "      fi_ht::sint tunedMaxNumCtasPerSeqKv = maxNumCtasPerSeqKv;")
sub(F, "          int const launchedClusters =", "          fi_ht::sint const launchedClusters =")
sub(F, "          int const residentSplitBudget = std::max(", "          fi_ht::sint const residentSplitBudget = std::max<fi_ht::sint>(")
sub(F, "              4, flashinfer::ceil_div(params.mMultiProcessorCount * 13 / 20, launchedClusters));",
    "              4, flashinfer::ceil_div<fi_ht::sint>(params.mMultiProcessorCount * 13 / 20, launchedClusters));")
sub(F, "          int const targetMaxNumCtasPerSeqKv =\n              std::min(maxNumCtasPerSeqKv, std::min(maxKvSplitsPerCgaCluster, residentSplitBudget));",
    "          fi_ht::sint const targetMaxNumCtasPerSeqKv = std::min<fi_ht::sint>(\n"
    "              maxNumCtasPerSeqKv, std::min<fi_ht::sint>(maxKvSplitsPerCgaCluster, residentSplitBudget));")
sub(F, "            int const targetTileSizePerCtaKv =", "            fi_ht::sint const targetTileSizePerCtaKv =")
sub(F, "                    ? flashinfer::ceil_div(params.mAttentionWindowSize,\n",
    "                    ? flashinfer::ceil_div<fi_ht::sint>(params.mAttentionWindowSize,\n")
sub(F, "      numCtasPerSeqKv = std::min(\n          tunedMaxNumCtasPerSeqKv,\n"
       "          std::max(1, int32_t(params.mMultiProcessorCount / (numCtasX * numCtasY * numCtasZ))));",
    "      numCtasPerSeqKv = std::min<fi_ht::sint>(\n          tunedMaxNumCtasPerSeqKv,\n"
    "          std::max<fi_ht::sint>(1, fi_ht::s32(params.mMultiProcessorCount / (numCtasX * numCtasY * numCtasZ))));")
sub(F, "      int totalNumCtas = numCtasX * numCtasZ * numCtasY;", "      fi_ht::sint totalNumCtas = numCtasX * numCtasZ * numCtasY;")
sub(F, "    int clusterDimX = selectKernelParams.mUses2CtaMma ? 2 : 1;\n    if (isCgaSmemReduction",
    "    fi_ht::sint clusterDimX = selectKernelParams.mUses2CtaMma ? 2 : 1;\n    if (isCgaSmemReduction")
sub(F, "    int totalNumCtas = numCtasX * numCtasZ * numCtasY;", "    fi_ht::sint totalNumCtas = numCtasX * numCtasZ * numCtasY;")
# MLA heuristics (compiled, not reached by the GQA calls traced here)
sub(F, "    int const maxNumCtasPerSeqKv = flashinfer::ceil_div(params.mMaxSeqLenKv, 256);",
    "    fi_ht::sint const maxNumCtasPerSeqKv = flashinfer::ceil_div(params.mMaxSeqLenKv, 256);")
sub(F, "    int const numCtas = static_cast<int32_t>(params.mBatchSize * params.mMaxSeqLenQ *",
    "    fi_ht::sint const numCtas = static_cast<fi_ht::s32>(params.mBatchSize * params.mMaxSeqLenQ *")
sub(F, "    int const numCtasPerSeqKv =\n        std::min(maxNumCtasPerSeqKv, std::max(1, int32_t(params.mMultiProcessorCount / numCtas)));",
    "    fi_ht::sint const numCtasPerSeqKv = std::min<fi_ht::sint>(\n"
    "        maxNumCtasPerSeqKv, std::max<fi_ht::sint>(1, fi_ht::s32(params.mMultiProcessorCount / numCtas)));")
sub(F, "    int const seqLenPerCtaKv = flashinfer::ceil_div(params.mMaxSeqLenKv, numCtasPerSeqKv);",
    "    fi_ht::sint const seqLenPerCtaKv = flashinfer::ceil_div(params.mMaxSeqLenKv, numCtasPerSeqKv);")
sub(F, "      int const effectiveSeqLenKv = std::min(params.mMaxSeqLenKv, params.mSparseMlaTopK);",
    "      fi_ht::sint const effectiveSeqLenKv = std::min<fi_ht::sint>(params.mMaxSeqLenKv, params.mSparseMlaTopK);")
sub(F, "      int const maxNumCtasPerSeqKv =\n          flashinfer::ceil_div(effectiveSeqLenKv, selectKernelParams.mTileSizeKv);",
    "      fi_ht::sint const maxNumCtasPerSeqKv =\n          flashinfer::ceil_div(effectiveSeqLenKv, selectKernelParams.mTileSizeKv);")
sub(F, "    int const groupedRows = params.mMaxSeqLenQ * params.mNumHeadsQPerKv;\n    int const baseNumCtas =\n",
    "    fi_ht::sint const groupedRows = params.mMaxSeqLenQ * params.mNumHeadsQPerKv;\n    fi_ht::sint const baseNumCtas =\n")
sub(F, "    int numTokensHeadsQ = params.mNumHeadsQPerKv * params.mMaxSeqLenQ;",
    "    fi_ht::sint numTokensHeadsQ = params.mNumHeadsQPerKv * params.mMaxSeqLenQ;")
# the concrete conditions first (same value): a trace then guards the batch only where the head dims hold
sub(F, "    bool isDsv3MinLatencyMode = params.mBatchSize == 1 && params.mMaxSeqLenQ >= 1 &&\n"
       "                                params.mMaxSeqLenQ <= 16 && params.mHeadDimQk == 576 &&\n"
       "                                params.mHeadDimV == 512;",
    "#ifdef FI_HT_TRACED\n"
    "    bool isDsv3MinLatencyMode = params.mHeadDimQk == 576 && params.mHeadDimV == 512 &&\n"
    "                                params.mBatchSize == 1 && params.mMaxSeqLenQ >= 1 &&\n"
    "                                params.mMaxSeqLenQ <= 16;\n"
    "#else\n"
    "    bool isDsv3MinLatencyMode = params.mBatchSize == 1 && params.mMaxSeqLenQ >= 1 &&\n"
    "                                params.mMaxSeqLenQ <= 16 && params.mHeadDimQk == 576 &&\n"
    "                                params.mHeadDimV == 512;\n"
    "#endif")
# the GQA spec-decode tile cost model (float): not traced yet
sub(F, "  void selectTileSizeQForGqaGeneration(RunnerParams const& params,\n"
       "                                       SelectKernelParams& selectKernelParams) const {\n",
    "  void selectTileSizeQForGqaGeneration(RunnerParams const& params,\n"
    "                                       SelectKernelParams& selectKernelParams) const {\n"
    "#ifdef FI_HT_TRACED\n"
    '    fi_ht::decline("the spec-decode tile cost model (generation at max_q > 1) is a later item");\n'
    "#else\n")
sub(F, "    // Apply the same sync to the committed kernelType; the probe above mutated only the copy.\n"
       "    syncGqaGenerationTraitsForKernelHash(params, selectKernelParams);\n  }\n",
    "    // Apply the same sync to the committed kernelType; the probe above mutated only the copy.\n"
    "    syncGqaGenerationTraitsForKernelHash(params, selectKernelParams);\n#endif  // FI_HT_TRACED\n  }\n")
# a cubin load: refused under a capture in the traced build
sub(F, "      if (findModuleIter == mModules.end()) {\n        // Load the module.\n",
    "      if (findModuleIter == mModules.end()) {\n        fi_ht::loading(kernelMeta.mFuncName);\n        // Load the module.\n")
sub(F, "  Data_type mDtypeQ, mDtypeK, mDtypeV, mDtypeOut;\n  int mNumEltsPerSageAttnBlkQ",
    "#ifdef FI_HT_TRACED\n public:\n"
    "  // the host-trace warm-up: the kernels a call of this class can select, by the members no trace value decides\n"
    "  std::vector<uint64_t> htCandidates(RunnerParams const& params) const {\n"
    "    std::vector<uint64_t> out;\n"
    "    for (auto const& [hash, index] : mKernelMetaMap) {\n"
    "      auto const& m = mKernelMeta[index];\n"
    "      if (m.mQkvLayout == static_cast<int>(params.mQkvLayout) && m.mHeadDimQk == params.mHeadDimQk &&\n"
    "          m.mHeadDimV == params.mHeadDimV &&\n"
    "          isContextKernel(static_cast<FmhaKernelType>(m.mKernelType)) == isContextKernel(params.mKernelType) &&\n"
    "          (m.mNumTokensPerPage == params.mNumTokensPerPage ||\n"
    "           m.mNumTokensPerPage == kDynamicNumTokensPerPageKernelKey)) {\n"
    "        out.push_back(hash);\n"
    "      }\n"
    "    }\n"
    "    return out;\n"
    "  }\n"
    "  char const* htName(uint64_t hash) const { return mKernelMeta[mKernelMetaMap.at(hash)].mFuncName; }\n"
    "  void htLoadNamed(std::string const& name) const {\n"
    "    for (auto const& [hash, index] : mKernelMetaMap) {\n"
    "      if (name == mKernelMeta[index].mFuncName) {\n"
    "        htLoad(hash);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  void htLoad(uint64_t hash) const {\n"
    "    if (mFunctions.count(hash)) {\n"
    "      return;\n"
    "    }\n"
    "    auto const metaIndex = mKernelMetaMap.at(hash);\n"
    "    auto const& kernelMeta = mKernelMeta[metaIndex];\n"
    "    std::string kernelName(kernelMeta.mFuncName);\n"
    "    CUmodule hmod{0};\n"
    "    auto it = mModules.find(kernelName);\n"
    "    if (it == mModules.end()) {\n"
    "      std::string cubin = getCubin(tllm_gen_fmha_cubin_path + \"/\" + kernelName + \".cubin\", kernelMeta.sha256);\n"
    "      if (cubin.empty()) {\n"
    "        throw std::runtime_error(\"Failed to load cubin for \" + kernelName);\n"
    "      }\n"
    "      cuErrCheck(cuModuleLoadData(&hmod, cubin.data()));\n"
    "      mModules[kernelName] = hmod;\n"
    "    } else {\n"
    "      hmod = it->second;\n"
    "    }\n"
    "    KernelInfo funcInfo;\n"
    "    funcInfo.mMetaInfoIndex = metaIndex;\n"
    "    cuErrCheck(cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName));\n"
    "    setupKernelSmem(funcInfo.mDeviceFunction, kernelMeta);\n"
    "    mFunctions[hash] = funcInfo;\n"
    "  }\n"
    "#endif  // FI_HT_TRACED\n"
    "  Data_type mDtypeQ, mDtypeK, mDtypeV, mDtypeOut;\n  int mNumEltsPerSageAttnBlkQ")

# ---------------------------------------------------------------- trtllm_fmha_kernel_launcher.cu
C = "csrc/trtllm_fmha_kernel_launcher.cu"
load(C)


def subp(old: str, new: str, count: int = 1) -> None:
    """A substitution in the paged launcher and bindings (ahead of the ragged launcher, which is not traced)."""
    sub(C, old, new, count, before="void trtllm_ragged_attention_launcher(")


subp("using tvm::ffi::Optional;\nusing tvm::ffi::Variant;\n",
    "#include <fi_ht.h>\n#ifndef FI_HT_TRACED  // the traced build's Optional and Variant: ht_trace.h\n"
    "using tvm::ffi::Optional;\nusing tvm::ffi::Variant;\n#endif\n")
subp("inline size_t getTrtllmGenMultiCtasKvCounterBytes(int64_t batch_size, int64_t num_qo_heads,\n"
       "                                                  int64_t sm_count) {\n"
       "  size_t const request_counter_slots =\n"
       "      static_cast<size_t>(batch_size) * static_cast<size_t>(num_qo_heads);\n"
       "  size_t const sm_counter_slots = static_cast<size_t>(sm_count);\n"
       "  size_t const num_semaphores =\n"
       "      round_up(std::max(request_counter_slots, sm_counter_slots), static_cast<size_t>(8));",
    "inline fi_ht::sz getTrtllmGenMultiCtasKvCounterBytes(fi_ht::s64 batch_size, int64_t num_qo_heads,\n"
    "                                                     int64_t sm_count) {\n"
    "  fi_ht::sz const request_counter_slots =\n"
    "      static_cast<fi_ht::sz>(batch_size) * static_cast<size_t>(num_qo_heads);\n"
    "  size_t const sm_counter_slots = static_cast<size_t>(sm_count);\n"
    "  fi_ht::sz const num_semaphores =\n"
    "      round_up(std::max<fi_ht::sz>(request_counter_slots, sm_counter_slots), static_cast<size_t>(8));")
subp("""void trtllm_paged_attention_launcher(
    void* out, void* out_scale_factor, void* query, void* key_cache, void* value_cache,
    void* workspace_buffer, void* multi_ctas_kv_counter_buffer, int64_t multi_ctas_kv_counter_size,
    int* block_tables, const void* k_block_scales_ptr, const void* v_block_scales_ptr,
    int* seq_lens, int* cum_seq_lens_q, int* cum_seq_lens_kv, float* attention_sinks, float* lse,
    Data_type q_data_type, Data_type kv_data_type, Data_type o_data_type,
    TllmPagedAttentionMode mode, int64_t batch_size, int64_t max_q_len, int64_t max_kv_len,
    int64_t num_pages_in_mem_pool, int64_t num_qo_heads, int64_t num_kv_heads, int64_t head_dim_qk,
    int64_t head_dim_vo, int64_t page_size, int64_t q_stride_tokens, int64_t q_stride_heads,
    int64_t kv_stride_keys_values, int64_t kv_stride_heads, int64_t kv_stride_batch,
    int64_t max_num_blocks_per_seq, double bmm1_scale, double bmm2_scale,
    const float* bmm1_scale_log2_ptr, const float* bmm2_scale_ptr, double o_sf_scale,
    int64_t o_sf_vec_size, int64_t o_sf_start_index, int64_t window_left, int64_t sum_seq_q,
    int64_t sparse_mla_top_k, void* sliding_window_kv_pool, int* sparse_mla_top_k_lens,
    bool has_sliding_window_kv_pool, float skip_softmax_threshold_scale_factor, bool skips_softmax,
    bool uses_shared_paged_kv_idx, bool enable_block_sparse_attention, int64_t sm_count,
    bool enable_pdl, int64_t workspace_size, int64_t k_sf_stride_heads, int64_t k_sf_stride_batch,
    int64_t v_sf_stride_heads, int64_t v_sf_stride_batch, bool is_causal, int64_t lse_stride_tokens,
    int64_t lse_stride_heads, int64_t bf16q_fp8kv_transform_mode, bool use_fp16_softmax,
    bool uses_spcompress, cudaStream_t stream) {""",
    """void trtllm_paged_attention_launcher(
    fi_ht::ptr<void> out, fi_ht::ptr<void> out_scale_factor, fi_ht::ptr<void> query,
    fi_ht::ptr<void> key_cache, fi_ht::ptr<void> value_cache, fi_ht::ptr<void> workspace_buffer,
    fi_ht::ptr<void> multi_ctas_kv_counter_buffer, int64_t multi_ctas_kv_counter_size,
    fi_ht::ptr<int> block_tables, fi_ht::ptr<const void> k_block_scales_ptr,
    fi_ht::ptr<const void> v_block_scales_ptr, fi_ht::ptr<int> seq_lens, fi_ht::ptr<int> cum_seq_lens_q,
    fi_ht::ptr<int> cum_seq_lens_kv, fi_ht::ptr<float> attention_sinks, fi_ht::ptr<float> lse,
    Data_type q_data_type, Data_type kv_data_type, Data_type o_data_type,
    TllmPagedAttentionMode mode, fi_ht::s64 batch_size, fi_ht::s64 max_q_len, fi_ht::s64 max_kv_len,
    int64_t num_pages_in_mem_pool, int64_t num_qo_heads, int64_t num_kv_heads, int64_t head_dim_qk,
    int64_t head_dim_vo, int64_t page_size, int64_t q_stride_tokens, int64_t q_stride_heads,
    int64_t kv_stride_keys_values, int64_t kv_stride_heads, int64_t kv_stride_batch,
    fi_ht::s64 max_num_blocks_per_seq, double bmm1_scale, double bmm2_scale,
    fi_ht::ptr<const float> bmm1_scale_log2_ptr, fi_ht::ptr<const float> bmm2_scale_ptr, double o_sf_scale,
    int64_t o_sf_vec_size, int64_t o_sf_start_index, int64_t window_left, fi_ht::s64 sum_seq_q,
    int64_t sparse_mla_top_k, fi_ht::ptr<void> sliding_window_kv_pool, fi_ht::ptr<int> sparse_mla_top_k_lens,
    bool has_sliding_window_kv_pool, float skip_softmax_threshold_scale_factor, bool skips_softmax,
    bool uses_shared_paged_kv_idx, bool enable_block_sparse_attention, int64_t sm_count,
    bool enable_pdl, fi_ht::s64 workspace_size, int64_t k_sf_stride_heads, int64_t k_sf_stride_batch,
    int64_t v_sf_stride_heads, int64_t v_sf_stride_batch, bool is_causal, fi_ht::s64 lse_stride_tokens,
    fi_ht::s64 lse_stride_heads, int64_t bf16q_fp8kv_transform_mode, bool use_fp16_softmax,
    bool uses_spcompress, cudaStream_t stream) {""")
subp("  size_t const counter_bytes =\n", "  fi_ht::sz const counter_bytes =\n")
subp("    runner_params.multiCtasKvCounterPtr = reinterpret_cast<int32_t*>(multi_ctas_kv_counter_buffer);",
     "    runner_params.multiCtasKvCounterPtr = fi_ht::ptr_cast<int32_t*>(multi_ctas_kv_counter_buffer);")
subp('    TVM_FFI_CHECK(reinterpret_cast<std::uintptr_t>(multi_ctas_kv_counter_buffer) % 16 == 0,',
    '    TVM_FFI_CHECK(fi_ht::aligned(multi_ctas_kv_counter_buffer, 16),')
subp("    TVM_FFI_CHECK(static_cast<size_t>(multi_ctas_kv_counter_size) >= counter_bytes,\n"
       '                  "trtllm-gen multi-CTA KV counter buffer is too small: got " +\n'
       "                      std::to_string(multi_ctas_kv_counter_size) + \" bytes, need \" +\n"
       "                      std::to_string(counter_bytes) + \" bytes\");",
     "    TVM_FFI_CHECK(static_cast<fi_ht::sz>(multi_ctas_kv_counter_size) >= counter_bytes,\n"
     '                  "trtllm-gen multi-CTA KV counter buffer is too small: got " +\n'
     "                      fi_ht::to_string(multi_ctas_kv_counter_size) + \" bytes, need \" +\n"
     "                      fi_ht::to_string(counter_bytes) + \" bytes\");")
# the counter buffer is FlashInfer's slice of the workspace sized by the batch: its size stays symbolic
subp("    fi_ht::ptr<void> multi_ctas_kv_counter_buffer, int64_t multi_ctas_kv_counter_size,",
     "    fi_ht::ptr<void> multi_ctas_kv_counter_buffer, fi_ht::s64 multi_ctas_kv_counter_size,")
subp("      multi_ctas_kv_counter_buffer.numel() * get_element_size(multi_ctas_kv_counter_buffer),",
     "      fi_ht::sym_numel(multi_ctas_kv_counter_buffer) * get_element_size(multi_ctas_kv_counter_buffer),", 2)
subp("    size_t const softmax_slots = static_cast<size_t>(num_qo_heads) *\n"
       "                                 static_cast<size_t>(batch_size) *\n"
       "                                 static_cast<size_t>(round_up(max_q_len, int64_t{256}));",
    "    fi_ht::sz const softmax_slots = static_cast<size_t>(num_qo_heads) *\n"
    "                                    static_cast<fi_ht::sz>(batch_size) *\n"
    "                                    static_cast<fi_ht::sz>(round_up(max_q_len, int64_t{256}));")
# the bindings: decode, then context
subp("  int sum_seq_q = query.size(0);\n  int num_qo_heads = query.size(1);\n  // the cum_seq_lens_q",
    "  fi_ht::sint sum_seq_q = fi_ht::sym_size(query, 0);\n  int num_qo_heads = query.size(1);\n  // the cum_seq_lens_q")
subp("  int* cum_seq_lens_q_ptr =\n      cum_seq_lens_q.has_value() ? static_cast<int*>(cum_seq_lens_q.value().data_ptr()) : nullptr;",
    "  fi_ht::ptr<int> cum_seq_lens_q_ptr =\n      cum_seq_lens_q.has_value() ? fi_ht::ptr_cast<int*>(cum_seq_lens_q.value().data_ptr()) : nullptr;")
subp("  int num_qo_heads = query.size(1);\n  int sum_seq_q = query.size(0);",
    "  int num_qo_heads = query.size(1);\n  fi_ht::sint sum_seq_q = fi_ht::sym_size(query, 0);")
subp("  int max_num_blocks_per_seq = block_tables.size(-1);", "  fi_ht::sint max_num_blocks_per_seq = fi_ht::sym_size(block_tables, -1);", 2)
subp("  const void* k_block_scales_ptr =", "  fi_ht::ptr<const void> k_block_scales_ptr =", 2)
subp("  const void* v_block_scales_ptr =", "  fi_ht::ptr<const void> v_block_scales_ptr =", 2)
subp("  void* output_sf_ptr =", "  fi_ht::ptr<void> output_sf_ptr =", 2)
subp("  float* attention_sinks_ptr = nullptr;", "  fi_ht::ptr<float> attention_sinks_ptr = nullptr;", 2)
subp("    attention_sinks_ptr = static_cast<float*>(attention_sinks.value().data_ptr());",
    "    attention_sinks_ptr = fi_ht::ptr_cast<float*>(attention_sinks.value().data_ptr());", 2)
subp("  float* lse_ptr = nullptr;", "  fi_ht::ptr<float> lse_ptr = nullptr;", 2)
subp("    lse_ptr = static_cast<float*>(lse.value().data_ptr());", "    lse_ptr = fi_ht::ptr_cast<float*>(lse.value().data_ptr());", 2)
subp("  int* sparse_mla_top_k_lens_ptr = nullptr;", "  fi_ht::ptr<int> sparse_mla_top_k_lens_ptr = nullptr;")
subp("    sparse_mla_top_k_lens_ptr = static_cast<int*>(top_k_lens.data_ptr());",
    "    sparse_mla_top_k_lens_ptr = fi_ht::ptr_cast<int*>(top_k_lens.data_ptr());")
subp("  float* bmm1_scale_log2_ptr =\n      maybe_bmm1_scale_log2_tensor.has_value()\n"
       "          ? static_cast<float*>(maybe_bmm1_scale_log2_tensor.value().data_ptr())\n          : nullptr;",
    "  fi_ht::ptr<float> bmm1_scale_log2_ptr =\n      maybe_bmm1_scale_log2_tensor.has_value()\n"
    "          ? fi_ht::ptr_cast<float*>(maybe_bmm1_scale_log2_tensor.value().data_ptr())\n          : nullptr;", 2)
subp("  float* bmm2_scale_ptr = maybe_bmm2_scale_tensor.has_value()\n"
       "                              ? static_cast<float*>(maybe_bmm2_scale_tensor.value().data_ptr())\n"
       "                              : nullptr;",
    "  fi_ht::ptr<float> bmm2_scale_ptr = maybe_bmm2_scale_tensor.has_value()\n"
    "                                         ? fi_ht::ptr_cast<float*>(maybe_bmm2_scale_tensor.value().data_ptr())\n"
    "                                         : nullptr;", 2)
subp("      static_cast<int*>(block_tables.data_ptr()), k_block_scales_ptr, v_block_scales_ptr,\n"
       "      static_cast<int*>(seq_lens.data_ptr()), cum_seq_lens_q_ptr,",
    "      fi_ht::ptr_cast<int*>(block_tables.data_ptr()), k_block_scales_ptr, v_block_scales_ptr,\n"
    "      fi_ht::ptr_cast<int*>(seq_lens.data_ptr()), cum_seq_lens_q_ptr,")
subp("      static_cast<int*>(block_tables.data_ptr()), k_block_scales_ptr, v_block_scales_ptr,\n"
       "      static_cast<int*>(seq_lens.data_ptr()),\n"
       "      /*cum_seq_lens_q=*/static_cast<int*>(cum_seq_lens_q.data_ptr()),\n"
       "      /*cum_seq_lens_kv=*/static_cast<int*>(cum_seq_lens_kv.data_ptr()),",
    "      fi_ht::ptr_cast<int*>(block_tables.data_ptr()), k_block_scales_ptr, v_block_scales_ptr,\n"
    "      fi_ht::ptr_cast<int*>(seq_lens.data_ptr()),\n"
    "      /*cum_seq_lens_q=*/fi_ht::ptr_cast<int*>(cum_seq_lens_q.data_ptr()),\n"
    "      /*cum_seq_lens_kv=*/fi_ht::ptr_cast<int*>(cum_seq_lens_kv.data_ptr()),")
subp("void trtllm_paged_attention_decode(\n    TensorView out, Optional<TensorView> out_scale_factor, TensorView query, TensorView key_cache,\n"
       "    TensorView value_cache, TensorView workspace_buffer, TensorView multi_ctas_kv_counter_buffer,\n"
       "    TensorView block_tables, TensorView seq_lens, int64_t max_q_len, int64_t max_kv_len,\n",
    "void trtllm_paged_attention_decode(\n    TensorView out, Optional<TensorView> out_scale_factor, TensorView query, TensorView key_cache,\n"
    "    TensorView value_cache, TensorView workspace_buffer, TensorView multi_ctas_kv_counter_buffer,\n"
    "    TensorView block_tables, TensorView seq_lens, fi_ht::s64 max_q_len, fi_ht::s64 max_kv_len,\n")
subp("    double o_sf_scale, int64_t o_sf_vec_size, int64_t o_sf_start_index, int64_t batch_size,\n"
       "    int64_t window_left, int64_t sparse_mla_top_k, int64_t sm_count, bool enable_pdl,\n"
       "    int64_t workspace_size, Optional<TensorView> attention_sinks,\n",
    "    double o_sf_scale, int64_t o_sf_vec_size, int64_t o_sf_start_index, fi_ht::s64 batch_size,\n"
    "    int64_t window_left, int64_t sparse_mla_top_k, int64_t sm_count, bool enable_pdl,\n"
    "    fi_ht::s64 workspace_size, Optional<TensorView> attention_sinks,\n")
subp("    Optional<bool> uses_shared_paged_kv_idx, Optional<TensorView> lse, int64_t lse_stride_tokens,\n"
       "    int64_t lse_stride_heads, bool enable_block_sparse_attention,\n",
    "    Optional<bool> uses_shared_paged_kv_idx, Optional<TensorView> lse, fi_ht::s64 lse_stride_tokens,\n"
    "    fi_ht::s64 lse_stride_heads, bool enable_block_sparse_attention,\n")
subp("void trtllm_paged_attention_context(\n    TensorView out, Optional<TensorView> out_scale_factor, TensorView query, TensorView key_cache,\n"
       "    TensorView value_cache, TensorView workspace_buffer, TensorView multi_ctas_kv_counter_buffer,\n"
       "    TensorView block_tables, TensorView seq_lens, int64_t max_q_len, int64_t max_kv_len,\n"
       "    Variant<double, ffi::Tensor> bmm1_scale, Variant<double, ffi::Tensor> bmm2_scale,\n"
       "    double o_sf_scale, int64_t o_sf_vec_size, int64_t o_sf_start_index, int64_t batch_size,\n"
       "    int64_t window_left, TensorView cum_seq_lens_q, TensorView cum_seq_lens_kv, int64_t sm_count,\n"
       "    bool enable_pdl, int64_t workspace_size, Optional<TensorView> attention_sinks,\n",
    "void trtllm_paged_attention_context(\n    TensorView out, Optional<TensorView> out_scale_factor, TensorView query, TensorView key_cache,\n"
    "    TensorView value_cache, TensorView workspace_buffer, TensorView multi_ctas_kv_counter_buffer,\n"
    "    TensorView block_tables, TensorView seq_lens, fi_ht::s64 max_q_len, fi_ht::s64 max_kv_len,\n"
    "    Variant<double, ffi::Tensor> bmm1_scale, Variant<double, ffi::Tensor> bmm2_scale,\n"
    "    double o_sf_scale, int64_t o_sf_vec_size, int64_t o_sf_start_index, fi_ht::s64 batch_size,\n"
    "    int64_t window_left, TensorView cum_seq_lens_q, TensorView cum_seq_lens_kv, int64_t sm_count,\n"
    "    bool enable_pdl, fi_ht::s64 workspace_size, Optional<TensorView> attention_sinks,\n")
subp("    Optional<TensorView> lse, int64_t lse_stride_tokens, int64_t lse_stride_heads) {\n",
    "    Optional<TensorView> lse, fi_ht::s64 lse_stride_tokens, fi_ht::s64 lse_stride_heads) {\n")
# the ragged and MLA bindings, the cubin loader and the TVM-FFI exports: FlashInfer's build only
sub(C, "void trtllm_ragged_attention_launcher(\n", "#ifndef FI_HT_TRACED  // not traced: ragged, sparse MLA\nvoid trtllm_ragged_attention_launcher(\n")
sub(C, "\nnamespace trtllm_cubin_loader {\n", "\n#endif  // FI_HT_TRACED\n#ifndef FI_HT_TRACED\nnamespace trtllm_cubin_loader {\n")
sub(C, "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_ragged_attention, trtllm_ragged_attention);\n",
    "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_ragged_attention, trtllm_ragged_attention);\n#endif  // FI_HT_TRACED\n")

# min/max of symbolic sizes: an expression in the traced build (fi_ht::min/max), not a guard on which side wins, as
# std::min/max's comparison would be; std::min/max in FlashInfer's own build. (The LSE block size stays a guard: a
# block dimension is the node's.)
for rel in (F, C):
    for fn, t in (("min", "sint"), ("max", "sint"), ("max", "sz")):
        n = files[rel].count(f"std::{fn}<fi_ht::{t}>(")
        files[rel] = files[rel].replace(f"std::{fn}<fi_ht::{t}>(", f"fi_ht::{fn}<fi_ht::{t}>(")

# no braced list builds a container anywhere in the patched files (the cudafe bug above): every one is parenthesized
BRACE_LIST = re.compile(r"\b(?:std::vector|std::array|std::initializer_list|c10::SmallVector|SmallVector|DimVector|SymDimVector)"
                        r"\s*<[^;{}()]*>\s*(?:\w+\s*)?\{")
for rel, text in files.items():
    for m in BRACE_LIST.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        sys.exit(f"{rel}:{line}: a braced container list ({m.group(0)!r}); write it parenthesized: T({{...}})")

for rel, text in files.items():
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
print(f"patched {len(files)} files -> {OUT}")
