#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/SymDynamic.h>
#include <ATen/cuda/host_trace/SymImage.h>

#include <ATen/native/cuda/VectorWidth.h>
#include <ATen/native/cuda/thread_constants.h>
#include <c10/cuda/CUDAException.h>

#include <cstring>

namespace at::cuda::host_trace {

namespace {
thread_local Recorder* current = nullptr;
thread_local std::vector<PendingScalar> pending;

// IntDivider<unsigned int>'s fields for divisor d: divisor, m1 (as the
// 4-byte field's signed bits), shift = ceil(log2(d)) through the trace's rows
template <class K>
void set_divider(Recorder& rec, Param<K>& p, at::cuda::detail::IntDivider<unsigned int>& dst, const c10::SymInt& d) {
  const c10::SymInt shift = rec.bit_length(d - 1);
  const c10::SymInt m1 = at::cuda::detail::int_divider_magic<c10::SymInt>(d, shift, [&](const c10::SymInt& s) { return rec.pow2(s); });
  p.set(dst.divisor, d);
  p.set(dst.m1, m1 - m1 / (int64_t(1) << 31) * (int64_t(1) << 32));
  p.set(dst.shift, shift);
}

// one kernel parameter's bytes, as the kernel receives them, and its symbolic fields
struct Image {
  std::vector<uint8_t> bytes;
  std::vector<Field> fields;
  std::vector<CpuScalarField> cpu_scalars;

  void put(size_t at, const void* src, size_t size) {
    TORCH_INTERNAL_ASSERT(at + size <= bytes.size());
    std::memcpy(bytes.data() + at, src, size);
  }
  // a SymInt member of `width` bytes at `at`: a constant written, a symbol recorded
  void set(size_t at, size_t width, const c10::SymInt& v, bool pointer) {
    if (auto c = v.maybe_as_int()) {
      const int64_t value = *c;
      put(at, &value, width);  // little-endian: the low bytes
    } else {
      fields.push_back({0, at, width, v, pointer});
    }
  }
  void pending_scalars_of(const void* object, size_t base) {
    for (const auto& s : pending) {
      if (s.object != object) {
        continue;
      }
      if (s.source.defined()) {
        cpu_scalars.push_back({0, base + s.offset, s.cls, s.source});
      } else {
        set(base + s.offset, s.width, s.bits, false);
      }
    }
  }
};

// OffsetCalculator<N>(dims, sizes, strides[, element sizes]) over SymInt, at `at` of image
template <int N>
void offset_calculator(Image& image, size_t at, const SymOffsets& c) {
  Recorder& rec = recorder_for("an OffsetCalculator");
  TORCH_CHECK(c.dims <= MAX_DIMS, "tensor has too many (>", MAX_DIMS, ") dims");
  TORCH_INTERNAL_ASSERT(c.strides.size() == static_cast<size_t>(N));
  Param<::OffsetCalculator<N>> p;
  auto& calc = p.value();
  p.set(calc.dims, c.dims);
  for (const auto i : c10::irange(c.dims)) {
    set_divider(rec, p, calc.sizes_[i], c.sizes[i]);
    for (const auto arg : c10::irange(N)) {
      p.set(calc.strides_[i][arg], c.strides[arg][i]);
    }
  }
  image.put(at, p.bytes.data(), p.bytes.size());
  for (Field f : p.fields) {
    f.offset += at;
    image.fields.push_back(std::move(f));
  }
}

void offset_calculator(Image& image, size_t at, int n, const SymOffsets& c) {
  switch (n) {
    case 1:
      return offset_calculator<1>(image, at, c);
    case 2:
      return offset_calculator<2>(image, at, c);
    case 3:
      return offset_calculator<3>(image, at, c);
    case 4:
      return offset_calculator<4>(image, at, c);
    case 5:
      return offset_calculator<5>(image, at, c);
    case 6:
      return offset_calculator<6>(image, at, c);
    case 7:
      return offset_calculator<7>(image, at, c);
    case 8:
      return offset_calculator<8>(image, at, c);
    default:
      TORCH_CHECK_NOT_IMPLEMENTED(false, "host_trace: an OffsetCalculator of ", n, " operands");
  }
}

void pointers(Image& image, size_t at, const c10::SymInt* data, size_t n) {
  for (const auto i : c10::irange(n)) {
    image.set(at + i * sizeof(char*), sizeof(char*), data[i], true);
  }
}

Image image_of(const ArgDesc& d) {
  Image image;
  image.bytes.assign(d.size, 0);
  switch (d.kind) {
    case ArgDesc::Scalar:
      TORCH_INTERNAL_ASSERT(d.size == 1 || d.size == 4 || d.size == 8);
      image.set(0, d.size, *static_cast<const c10::SymInt*>(d.value), false);
      break;
    case ArgDesc::Pointers:
      pointers(image, 0, static_cast<const c10::SymInt*>(d.value), d.n);
      break;
    case ArgDesc::Offsets:
      offset_calculator(image, 0, static_cast<int>(d.n), *static_cast<const SymOffsets*>(d.value));
      break;
    case ArgDesc::Bytes:
      image.put(0, d.value, d.size);
      image.pending_scalars_of(d.object, 0);
      break;
    case ArgDesc::Strided:
      image.put(0, d.value, d.size);
      image.pending_scalars_of(d.object, d.f_offset);
      pointers(image, d.data_offset, d.data, d.n);
      offset_calculator(image, d.calc_offset, static_cast<int>(d.n), *d.offsets);
      break;
  }
  return image;
}
} // namespace

Recorder* current_recorder() {
  return current;
}

RecorderGuard::RecorderGuard(Recorder* rec)
    : previous_(std::exchange(current, rec)), previous_symbolic_(at::detail::set_symbolic_iterators(rec != nullptr)) {}

RecorderGuard::~RecorderGuard() {
  current = previous_;
  at::detail::set_symbolic_iterators(previous_symbolic_);
}

Recorder& recorder_for(const char* what) {
  TORCH_CHECK_NOT_IMPLEMENTED(current != nullptr, "host_trace: ", what, " of a symbolic TensorIterator outside a host trace's recorder");
  return *current;
}

std::vector<PendingScalar>& pending_scalars() {
  return pending;
}

const char jitted_scalar_slot = 0;

ScalarScope::ScalarScope(const void* object, const void* member, char cls, const TensorBase& source) {
  const auto offset = static_cast<size_t>(static_cast<const char*>(member) - static_cast<const char*>(object));
  pending.push_back({object, offset, cls, source});
}

ScalarScope::ScalarScope(const void* object, const void* member, size_t width, c10::SymInt bits) {
  const auto offset = static_cast<size_t>(static_cast<const char*>(member) - static_cast<const char*>(object));
  pending.push_back({object, offset, 0, TensorBase(), width, std::move(bits)});
}

ScalarScope::~ScalarScope() {
  pending.pop_back();
}

c10::SymInt float_bits(const c10::SymInt& v) {
  // f32_div takes a numerator >= 0; a negative float's bits are its
  // magnitude's with the sign bit set
  Recorder& rec = recorder_for("a symbolic number's float");
  const c10::SymBool negative = v.sym_lt(0);
  const c10::SymInt bits = rec.f32_div(rec.select(negative, -v, v), 1);
  return rec.select(negative, bits - (int64_t(1) << 31), bits);
}

void unsupported_symbol_member(const char* type) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "host_trace: a symbolic number in a functor member of type ", type, ", which a replay cannot fill");
}

void unsupported_scalar_member(const char* type) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "host_trace: a host buffer's value in a functor member of type ", type, ", which a replay cannot fill");
}

c10::SymInt operand_address(const TensorIteratorBase& iter, int arg) {
  return recorder_for("an operand's address").data_ptr(iter.tensor_base(arg)) + iter.sym_data_offset(arg);
}

SymOffsets sym_offsets(const TensorIteratorBase& iter, int first, int n, bool element_strides) {
  SymOffsets c;
  c.dims = iter.ndim();
  const auto shape = iter.sym_shape();
  c.sizes.assign(shape.begin(), shape.end());
  for (const auto i : c10::irange(n)) {
    auto& strides = c.strides.emplace_back();
    for (const auto& st : iter.sym_strides(first + i)) {
      strides.push_back(element_strides ? st / iter.element_size(first + i) : st);
    }
  }
  return c;
}

ArgDesc offset_calculator_desc(int n, const SymOffsets& c) {
  ArgDesc d;
  d.kind = ArgDesc::Offsets;
  d.n = static_cast<uint32_t>(n);
  d.value = &c;
  switch (n) {
#define OFFSET_CALCULATOR_SIZE(N)              \
  case N:                                      \
    d.size = sizeof(::OffsetCalculator<N>);    \
    d.align = alignof(::OffsetCalculator<N>);  \
    break;
    OFFSET_CALCULATOR_SIZE(1)
    OFFSET_CALCULATOR_SIZE(2)
    OFFSET_CALCULATOR_SIZE(3)
    OFFSET_CALCULATOR_SIZE(4)
    OFFSET_CALCULATOR_SIZE(5)
    OFFSET_CALCULATOR_SIZE(6)
    OFFSET_CALCULATOR_SIZE(7)
    OFFSET_CALCULATOR_SIZE(8)
#undef OFFSET_CALCULATOR_SIZE
    default:
      TORCH_CHECK_NOT_IMPLEMENTED(false, "host_trace: an OffsetCalculator of ", n, " operands");
  }
  return d;
}

c10::SmallVector<c10::SymInt, SymFacts::max_tensors> SymFacts::data(const TensorIteratorBase& iter) const {
  c10::SmallVector<c10::SymInt, max_tensors> d;
  for (const auto i : c10::irange(ntensors)) {
    d.push_back(operand_address(iter, i));
  }
  return d;
}

// memory::can_vectorize_up_to<func_t>: the vector width each operand's
// address allows at its type's size (aligned_vector<T, vec>'s alignment)
int SymFacts::can_vectorize(c10::ArrayRef<c10::SymInt> data) const {
  int result = at::native::vector_width_for(sizes[0], data[0]);
  for (const auto i : c10::irange(1, ntensors)) {
    result = std::min(result, at::native::vector_width_for(sizes[i], data[i]));
  }
  return result;
}

// needs_dynamic_casting<func_t>::check
bool SymFacts::needs_dynamic_casting(const TensorIteratorBase& iter) const {
  for (const auto i : c10::irange(arity)) {
    if (iter.input_dtype(i) != types[i + 1]) {
      return true;
    }
  }
  return types[0] != ScalarType::Undefined && iter.dtype(0) != types[0];
}

namespace {

ArgDesc functor_desc(const SymFacts& f) {
  ArgDesc d = bytes_desc(f.functor, f.functor_size, f.functor);
  d.align = f.functor_align;
  return d;
}

const uint8_t zeros[16] = {};

ArgDesc empty_desc(uint32_t size) {
  TORCH_INTERNAL_ASSERT(size <= sizeof(zeros));
  ArgDesc d = bytes_desc(zeros, size);
  d.align = 1;
  return d;
}

const void* kernel_of(const SymFacts& f, SymFacts::Kernel k) {
  TORCH_INTERNAL_ASSERT(f.kernels[k] != nullptr, "host_trace: no kernel of route ", static_cast<int>(k), " for this functor");
  return f.kernels[k];
}

} // namespace

void SymFacts::record_vectorized(int vec_size, bool sm107, const c10::SymInt& grid, const c10::SymInt& n, c10::ArrayRef<c10::SymInt> data) const {
  const ArgDesc count = int_desc(n);
  const ArgDesc pointers = pointers_desc(data.data(), data.size());
  if (vec_size == 1) {
    // launch_vectorized_kernel's unrolled kernel of trivial offset calculators, no casts
    const ArgDesc args[] = {count, functor_desc(*this), pointers, empty_desc(trivial_in_size), empty_desc(trivial_out_size), empty_desc(plain_load_size), empty_desc(plain_store_size)};
    return record_kernel(kernel_of(*this, Unrolled), args, std::size(args), grid, num_threads(), 0);
  }
  const Kernel k = vec_size == 8 ? (sm107 ? Vec8Sm107 : Vec8) : vec_size == 4 ? Vec4 : Vec2;
  TORCH_INTERNAL_ASSERT(vec_size == 8 || vec_size == 4 || vec_size == 2, "Unexpected vectorization size");
  const ArgDesc args[] = {count, functor_desc(*this), pointers};
  record_kernel(kernel_of(*this, k), args, std::size(args), grid, num_threads(), 0);
}

void SymFacts::record_unrolled_cast(const c10::SymInt& grid, const c10::SymInt& n, c10::ArrayRef<c10::SymInt> data, const TensorIteratorBase& iter) const {
  std::vector<uint8_t> load, store;
  cast_params(iter, load, store);
  ArgDesc l = bytes_desc(load.data(), load.size());
  l.align = load_align;
  ArgDesc s = bytes_desc(store.data(), store.size());
  s.align = store_align;
  const ArgDesc args[] = {int_desc(n), functor_desc(*this), pointers_desc(data.data(), data.size()), empty_desc(trivial_in_size), empty_desc(trivial_out_size), l, s};
  record_kernel(kernel_of(*this, UnrolledCast), args, std::size(args), grid, num_threads(), 0);
}

void SymFacts::record_strided(bool cast, const c10::SymInt& grid, const c10::SymInt& n, c10::ArrayRef<c10::SymInt> data, const TensorIteratorBase& iter) const {
  // StridedOp / StridedCastOp{data, [dtypes,] offset_calc, f}: the functor's
  // bytes (and the operands' dtypes) in place, data and offsets symbolic
  const Layout& layout = cast ? strided_cast : strided;
  std::vector<uint8_t> raw(layout.size, 0);
  TORCH_INTERNAL_ASSERT(layout.f + functor_size <= raw.size());
  std::memcpy(raw.data() + layout.f, functor, functor_size);
  if (cast) {
    for (const auto i : c10::irange(ntensors)) {
      const ScalarType t = iter.dtype(i);
      std::memcpy(raw.data() + layout.dtypes + i * sizeof(ScalarType), &t, sizeof(ScalarType));
    }
  }
  const SymOffsets offsets = sym_offsets(iter, 0, ntensors, /*element_strides=*/false);
  ArgDesc op;
  op.kind = ArgDesc::Strided;
  op.size = layout.size;
  op.align = layout.align;
  op.n = static_cast<uint32_t>(ntensors);
  op.value = raw.data();
  op.object = functor;
  op.f_offset = layout.f;
  op.data = data.data();
  op.offsets = &offsets;
  op.data_offset = layout.data;
  op.calc_offset = layout.calc;
  const ArgDesc args[] = {int_desc(n), op};
  record_kernel(kernel_of(*this, cast ? StridedCast : Strided), args, std::size(args), grid, 128, 0);
}

void record_kernel(const void* kernel, const ArgDesc* args, size_t n, SymDim3 grid, SymDim3 block, int64_t smem) {
  cudaFunction_t function;
  C10_CUDA_CHECK(cudaGetFuncBySymbol(&function, kernel));
  record_function(function, args, n, std::move(grid), std::move(block), smem);
}

void record_function(cudaFunction_t function, const ArgDesc* args, size_t n, SymDim3 grid, SymDim3 block, int64_t smem) {
  Recorder& rec = recorder_for("a kernel launch");
  KernelRecord r;
  r.function = function;
  size_t end = 0;
  for (const auto i : c10::irange(n)) {
    Image image = image_of(args[i]);
    end = (end + args[i].align - 1) / args[i].align * args[i].align;
    r.offsets.push_back(end);
    end += image.bytes.size();
    for (Field f : image.fields) {
      f.param = i;
      r.fields.push_back(std::move(f));
    }
    for (CpuScalarField f : image.cpu_scalars) {
      f.param = i;
      r.cpu_scalars.push_back(std::move(f));
    }
    r.params.push_back(std::move(image.bytes));
  }
  r.grid = std::move(grid);
  r.block = std::move(block);
  r.smem = smem;
  rec.launches.push_back(std::move(r));
  rec.on_launch();
}

void put_offset_calculator(ParamImage& image, size_t at, int n, const SymOffsets& c) {
  Image calc;
  calc.bytes.assign(offset_calculator_desc(n, c).size, 0);
  offset_calculator(calc, 0, n, c);
  image.put(at, calc.bytes.data(), calc.bytes.size());
  for (Field f : calc.fields) {
    f.offset += at;
    image.fields.push_back(std::move(f));
  }
}

void record_images(const void* kernel, std::vector<ParamImage> params, SymDim3 grid, SymDim3 block, c10::SymInt smem) {
  cudaFunction_t function;
  C10_CUDA_CHECK(cudaGetFuncBySymbol(&function, kernel));
  record_images(function, std::move(params), std::move(grid), std::move(block), std::move(smem));
}

void record_images(cudaFunction_t function, std::vector<ParamImage> params, SymDim3 grid, SymDim3 block, c10::SymInt smem) {
  Recorder& rec = recorder_for("a kernel launch");
  KernelRecord r;
  r.function = function;
  size_t end = 0;
  for (auto& p : params) {
    end = (end + p.align - 1) / p.align * p.align;
    r.offsets.push_back(end);
    end += p.bytes.size();
    for (Field f : p.fields) {
      f.param = r.params.size();
      r.fields.push_back(std::move(f));
    }
    r.params.push_back(std::move(p.bytes));
  }
  r.grid = std::move(grid);
  r.block = std::move(block);
  r.smem = std::move(smem);
  rec.launches.push_back(std::move(r));
  rec.on_launch();
}

} // namespace at::cuda::host_trace
#endif
