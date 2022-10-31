#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>

#include <iostream>

namespace at {
namespace cuda {

MempoolId_t graph_pool_handle() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  // uuid count starts at 1. 0 is reserved to mean "wasn't set by graph_pool_handle".
  static std::atomic<CaptureId_t> uuid{1};
  // Sets just the second value, to distinguish it from MempoolId_ts created from
  // cudaStreamGetCaptureInfo id_s in capture_begin.
  return {0, uuid++};
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 and not yet supported on ROCM");
  return {0, 0};
#endif
}

/**
 * Note [CUDA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native CUDA ops (like RNG ops with
 *     CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [CUDA Graph-safe RNG states] in CUDAGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write capture
 *     bindings naively in an extension, they likely won't interact with the native
 *     ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with CUDA graph capture] in CUDACachingAllocator.cpp
 * describes memory management for captures.
 */

CUDAGraph::CUDAGraph()
  // CUDAStreams may not be default-constructed.
  : capture_stream_(at::cuda::getCurrentCUDAStream()) {
#if (defined(CUDA_VERSION) && CUDA_VERSION < 11000) || defined(USE_ROCM)
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 and not yet supported on ROCM");
#endif
}

void CUDAGraph::capture_begin(MempoolId_t pool/*=0*/) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  TORCH_CHECK(!has_graph_exec_,
              "This CUDAGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  // For now, a CUDAGraph instance only accommodates the default generator on the device that's
  // current when capture begins. If any op in the captured region uses a non-default generator,
  // or a generator on another device, the offending generator will throw an error.
  // These restrictions simplify CUDAGraph, but could be relaxed in the future:
  // in principle, the underlying Cuda calls do permit cross-device ops to be captured.
  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());

  auto options = TensorOptions().device(at::kCUDA).dtype(at::kLong);
  seed_extragraph_ = at::empty({1}, options);
  offset_extragraph_ = at::empty({1}, options);

  seed_extragraph_.fill_(int64_t(gen->current_seed()));
  gen->capture_prologue(seed_extragraph_.data_ptr<int64_t>(), offset_extragraph_.data_ptr<int64_t>());

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_gen_ = gen;
  capture_dev_ = c10::cuda::current_device();

  // cudaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe CUDA API calls during capture.  See
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeGlobal));

  // Stashes the current capture's uuid.
  cudaStreamCaptureStatus status;
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &id_));
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);

  // Ensures uuid count starts at 1. 0 is reserved to mean "not set by cudaStreamGetCaptureInfo".
  // (But how do we know GetCaptureInfo never sets id_ to 0? Because that's the current behavior,
  // and I asked cuda devs to keep it that way, and they agreed.)
  TORCH_INTERNAL_ASSERT(id_ > 0);
  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be nonzero.
    // If pool was created by graph_pool_handle, second should be nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Use our own id_ as our mempool_id_.
    // Sets just the first value, to distinguish it from MempoolId_ts created by graph_pool_handle().
    mempool_id_ = {id_, 0};
  }

  // When CUDACachingAllocator allocates while a capture is underway, it calls cudaStreamGetCaptureInfo
  // to get the current stream's capture id, if any. Here we tell CUDACachingAllocator: if the stream
  // has a capture id matching this graph's id_, use the private pool mempool_id_ identifies.
  //
  // There's a small chance of a bad allocation here if another thread launches a kernel on
  // capture_stream_ between the call to cudaStreamBeginCapture above and the call to
  // notifyCaptureBegin below.
  // But I don't think we need to worry about it because that use case makes no sense:
  // The user has no business launching kernels on capture_stream_ from another thread
  // while calling capture_begin. They'll have no idea if their side thread's
  // kernel will end up as part of the capture or not.
  c10::cuda::CUDACachingAllocator::notifyCaptureBegin(capture_dev_, id_, mempool_id_);
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 and not yet supported on ROCM");
#endif
}

void CUDAGraph::capture_end() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  c10::cuda::CUDACachingAllocator::notifyCaptureAboutToEnd(capture_dev_, id_);

  AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));
  TORCH_CHECK(graph_ != NULL, "Invalid capture.");
  has_graph_ = true;

  c10::cuda::CUDACachingAllocator::notifyCaptureEnded(capture_dev_, id_);

  // In typical graph usage some tensors (e.g. the tensors used for graph IO) are not freed
  // between replays.
  // If Pytorch compiles and runs with a CUDA 11.4+ toolkit, there's a chance the allocator backend
  // is cudaMallocAsync.
  // cudaMallocAsync is generally graph-safe, but if some tensors are not freed between replays,
  // the graph's internal bookkeeping requires that we instantiate with
  // cudaGraphInstantiateFlagAutoFreeOnLaunch. See
  // cudaGraphLaunch
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597
  // cudaGraphInstantiateWithFlags
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1ga2c652a24ba93e52b99a47bec0888233
#if CUDA_VERSION >= 11040
  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
#endif
    // Trailing NULL, NULL, 0 arguments were recommended by Cuda driver people,
    // who prefer not to report error message through these arguments moving forward
    // (they prefer return value, or errors on api calls internal to the capture)
    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
    // AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
#if CUDA_VERSION >= 11040
  } else {
    // AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_,
    //                                             graph_,
    //                                             cudaGraphInstantiateFlagAutoFreeOnLaunch));
    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuGraphInstantiateWithFlags(&graph_exec_,
                                            graph_,
                                            CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH));
}
#endif

  has_graph_exec_ = true;

  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());
  TORCH_CHECK(gen == capture_gen_,
              "Default CUDA RNG generator on current device at capture end "
              "is different from default generator on current device "
              "when capture began");
  wholegraph_increment_ = gen->capture_epilogue();

  // Now that we've instantiated graph_ into graph_exec_,
  // we don't need graph_ anymore.
  if (!preserve_graph_) {
    AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 and not yet supported on ROCM");
#endif
}

void CUDAGraph::replay() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::replay without a preceding successful capture.");

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  // Just like any RNG consumer kernel!
  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());
  PhiloxCudaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(wholegraph_increment_);
  }
  seed_extragraph_.fill_(int64_t(gen->current_seed()));
  offset_extragraph_.fill_(int64_t(rng_engine_inputs.offset_.val));

  // graph_exec_ may be replayed in any stream.
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));
  // AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));

  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
    // Workaround for bug in libcuda.so that causes replayed graphs with
    // certain topologies to be corrupted (kernels elided, internal syncs
    // ignored) when replayed back to back without a sync in between.
    // The bug is fixed in CUDA 11.4+.
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 and not yet supported on ROCM");
#endif
}

void CUDAGraph::reset() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  // I'd prefer these checks throw exceptions, not print warnings,
  // but the destructor calls reset(), and at least one CI build
  // refuses to compile with a throwing destructor.
  //
  // Instead of calling reset() in the destructor to clean up, I could
  // call reset() in the __del__ method of a thin Python wrapper,
  // in which case reset would be allowed to throw exceptions.
  // But Stackoverflow does not like user-defined __del__.
  // __del__ prevents Graph instances from EVER being garbage collected
  // if they participate in a reference cycle.
  // And exceptions thrown in __del__ only print a warning anyway.
  //
  // Calling reset() in the C++ destructor, with warnings instead of exceptions
  // if calls fail, is the compromise we chose.
  //
  // If capture_begin, the capture, or capture_end failed at some point, this CUDAGraph, the generator,
  // and the allocator could end up in all kinds of weird states depending where failure occurred.
  // If the user catches the failure exception in a script, or is running in REPL or (god forbid)
  // a Juptyer notebook, I don't see an easy way for reset() to gracefully fix all such possible error states.
  if (has_graph_ || has_graph_exec_) {
    // notifyCaptureDestroy may throw. How should we handle this?
    c10::cuda::CUDACachingAllocator::notifyCaptureDestroy(capture_dev_, mempool_id_);
  }
  if (has_graph_) {
    // C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
  }
  if (has_graph_exec_) {
    // C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 and not yet supported on ROCM");
#endif
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t CUDAGraph::pool() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::pool() without a preceding successful capture.");
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 and not yet supported on ROCM");
#endif
  return mempool_id_;
}

inline std::vector<cudaGraphNode_t> CUDAGraph::get_nodes(cudaGraph_t cuda_graph) {
  size_t numNodes;
  AT_CUDA_CHECK(cudaGraphGetNodes(cuda_graph, static_cast<cudaGraphNode_t*>(nullptr), &numNodes));
  if (numNodes == 0)
    return std::vector<cudaGraphNode_t>();
  std::vector<cudaGraphNode_t> graphNodes(numNodes);
  AT_CUDA_CHECK(cudaGraphGetNodes(cuda_graph, graphNodes.data(), &numNodes));
  return graphNodes;
}

inline std::string CUDAGraph::get_node_info(const cudaGraphNode_t node) {
  std::stringstream ss;

  cudaGraphNodeType pType;
  AT_CUDA_CHECK(cudaGraphNodeGetType(node, &pType));
  printf("Graph type = %i; ", pType);

  switch (pType) {
    case cudaGraphNodeTypeKernel: {
      CUDA_KERNEL_NODE_PARAMS kparams = {0};
      (at::globalContext().getNVRTC().cuGraphKernelNodeGetParams(node, &kparams));

      ss << "GPUKernel@" << kparams.func;
      ss << "<<<gridDim=(" << kparams.gridDimX << ", "<< kparams.gridDimY << ", " << kparams.gridDimZ << "), "
         << "blockDim=(" << kparams.blockDimX << ", "<< kparams.blockDimY << ", "<< kparams.blockDimZ << ")>>>";
      ss << "(";

      // we'll need to get exact number of parameters
      int64_t address_start = 1024;
      for(int64_t i=0; (int64_t)kparams.kernelParams[i] > address_start; i++) {
        void *ptr = *(void**)kparams.kernelParams[i];
        ss << reinterpret_cast<int64_t>(ptr) << ", ";
      }

      ss << ")";
      if (kparams.sharedMemBytes != 0)
        ss << ", dynSharedMemBytes=" << kparams.sharedMemBytes;
    } break;
    case cudaGraphNodeTypeMemcpy: {
      cudaMemcpy3DParms mparams = {};
      AT_CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node, &mparams));

      // If memcpy is seen, return without setting up runnable executor
      switch (mparams.kind) {
        case cudaMemcpyHostToHost:
          ss << "Host->Host ";
          break;
        case cudaMemcpyHostToDevice:
          ss << "Host->Device ";
          break;
        case cudaMemcpyDeviceToHost:
          ss << "Device->Host ";
          break;
        case cudaMemcpyDeviceToDevice:
          ss << "Device->Device ";
          break;
        default:
          break;
      }
      ss << "Memcpy";
    } break;
    case cudaGraphNodeTypeMemset: {
      cudaMemsetParams mparams = {};
      AT_CUDA_CHECK(cudaGraphMemsetNodeGetParams(node, &mparams));
      if (mparams.height == 1 && mparams.elementSize == 1) {
        ss << "cudaMemset(devPtr=" << mparams.dst << ", value=" << mparams.value
           << ", count=" << mparams.width << ")";
      } else {
        if (mparams.elementSize == 1)
          ss << "cudaMemset2D";
        else
          ss << "MemSet<elemBytes=" << mparams.elementSize << ">";
        ss << "(devPtr=" << mparams.dst << ", pitch=" << mparams.pitch
           << ", value=" << mparams.value << ", width=" << mparams.width
           << ", height=" << mparams.height << ")";
      }
    } break;
    case cudaGraphNodeTypeHost:
      ss << "Host (executable) node";
      break;
    case cudaGraphNodeTypeGraph:
      ss << "Node which executes an embedded graph";
      break;
    case cudaGraphNodeTypeEmpty:
      ss << "Empty (no-op) node";
      break;
    default:
      ss << "Unknown/Invalid node type " << pType;
  }

  return ss.str();
}


template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    fprintf(
        stderr,
        "CUDA error at %s:%d code=%d: %s \n",
        file,
        line,
        static_cast<unsigned int>(result),
        cudaGetErrorString(result));
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

void CUDAGraph::update_params(std::vector<Tensor> old_params, std::vector<Tensor> new_params) {
  TORCH_CHECK(preserve_graph_ && has_graph_ && has_graph_exec_);

  std::vector<cudaGraphNode_t> nodes = get_nodes(graph_);

  for(auto node : nodes) {
    TORCH_CHECK(node != NULL);
    std::cout<<get_node_info(node) << '\n';
  }
  // TODO - only handling one node right now
  auto node = nodes[0];

  CUDA_KERNEL_NODE_PARAMS kparams = {0};
  (at::globalContext().getNVRTC().cuGraphKernelNodeGetParams(node, &kparams));

  // Copies over block, grid values but still need to copy over kernelParmas
  CUDA_KERNEL_NODE_PARAMS kparams_copy = kparams;

  // Cudagraphs internally will call free on this pointer - using std::vector
  // here gives double free error. TODO - malloc right number of parmeters
  int64_t * vec = reinterpret_cast<int64_t*>(malloc(4 * sizeof(int64_t)));
  kparams_copy.kernelParams = reinterpret_cast<void**>(vec);

  // Copy over old parameters
  // TODO - right number of parameters to copy
  for(int64_t i=0; i < 4; i++) {
    vec[i] = reinterpret_cast<int64_t>(kparams.kernelParams[i]);
  }

  // Replace old data pointers with new data pointers. The data pointers
  // are passed in as a reference to an address which stores the data pointers,
  // so we allocate a new vector to store the values and pass pointers to it in.
  // TODO - right number of params.
  std::vector<int64_t> vec_new(6);
  vec_new[0] = reinterpret_cast<int64_t>(new_params[0].data_ptr());
  kparams_copy.kernelParams[0] = (&vec_new[0]);
  vec_new[1] = reinterpret_cast<int64_t>(new_params[1].data_ptr());
  kparams_copy.kernelParams[1] = (&vec_new[1]);


  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuGraphKernelNodeSetParams(node, &kparams_copy));
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuGraphExecKernelNodeSetParams(graph_exec_, node, &kparams_copy));
  cudaGraphExecUpdateResult updateResult;
  cudaGraphNode_t errorNode;

  // First we try to update the graph as this is much cheaper than re-instantiation
  checkCudaErrors(cudaGraphExecUpdate(graph_exec_, graph_, &errorNode, &updateResult));
  if (graph_exec_ == nullptr || updateResult != cudaGraphExecUpdateSuccess) {
    std::cout << " UNSUCCESSFUL BRANCH \n";

    // TODO: need to handle better
    // look at https://cs.github.com/hummingtree/cuda-graph-with-dynamic-parameters/blob/5a457dcd44d499e22f7cd34f420faf17e70fd994/gpu_graph.cpp?q=ExecUpdate#L50
    TORCH_CHECK(false);
    // The update is unsuccessful, need to re-instantiate
    AT_CUDA_CHECK(cudaGetLastError()); // <- Clear the error state
    if (graph_exec_ != nullptr) {
      AT_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
    }
    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
  }
}

CUDAGraph::~CUDAGraph() {
  reset();
}

} // namespace cuda
} // namespace at
