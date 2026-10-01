#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/TensorIteratorSym.h>
#include <ATen/detail/TensorIteratorBuild.h>

#include <type_traits>
#include <vector>

using namespace at;
namespace ti_build = at::detail::ti_build;

namespace {

template <typename T>
std::vector<int64_t> to_ints(c10::ArrayRef<T> xs) {
  std::vector<int64_t> r;
  for (const auto& x : xs) {
    if constexpr (std::is_same_v<T, c10::SymInt>) {
      r.push_back(x.expect_int());
    } else {
      r.push_back(x);
    }
  }
  return r;
}

template <typename T>
struct Operand {
  explicit Operand(TensorBase t, bool is_output = false) : tensor(std::move(t)), is_output(is_output) {
    if (tensor.defined()) {
      target_dtype = current_dtype = tensor.scalar_type();
    }
  }
  const TensorBase& tensor_base() const {
    return tensor;
  }
  bool is_type_defined() const {
    return target_dtype != ScalarType::Undefined;
  }

  TensorBase tensor;
  SmallVector<T, 6> stride_bytes;
  ScalarType target_dtype = ScalarType::Undefined;
  ScalarType current_dtype = ScalarType::Undefined;
  bool is_output = false;
  bool will_resize = false;
};

struct Layout {
  std::vector<int64_t> shape;
  std::vector<std::vector<int64_t>> strides;
  std::vector<int64_t> out_sizes;
  std::vector<int64_t> out_strides;
};

// TensorIteratorBase::build's shape/stride steps for an op with one fresh
// output, on T = int64_t or c10::SymInt
template <typename T>
Layout build(const std::vector<Tensor>& inputs) {
  using Shape = std::conditional_t<std::is_same_v<T, c10::SymInt>, SymDimVector, DimVector>;
  std::vector<Operand<T>> operands;
  operands.emplace_back(TensorBase(), /*is_output=*/true);
  operands[0].target_dtype = inputs[0].scalar_type();
  for (const auto& t : inputs) {
    operands.emplace_back(t);
  }
  auto set_output = [&](int i, c10::ArrayRef<T> sizes, c10::ArrayRef<T> strides, std::optional<MemoryFormat> memory_format) {
    auto& op = operands[i];
    if (op.tensor.defined()) {
      return;
    }
    auto options = inputs[0].options();
    op.tensor = strides.empty() ? at::empty(to_ints(sizes), options.memory_format(memory_format)) : at::empty_strided(to_ints(sizes), to_ints(strides), options);
    op.current_dtype = op.target_dtype;
  };

  Shape shape;
  DimVector perm;
  bool all_ops_same_shape = false, all_ops_are_scalars = false, has_coalesced_dimensions = false;
  ti_build::compute_shape(shape, operands, /*resize_outputs=*/true, all_ops_same_shape, all_ops_are_scalars);
  auto setup_type = ti_build::compute_fast_setup_type<T>(operands, /*is_reduction_=*/false, all_ops_same_shape, /*enforce_linear_iteration_=*/false);
  if (!ti_build::fast_set_up(setup_type, shape, operands, /*num_outputs_=*/1, has_coalesced_dimensions, set_output)) {
    ti_build::compute_strides(shape, operands, /*static_shape=*/false);
    ti_build::reorder_dimensions(shape, perm, operands, /*is_reduction_=*/false, /*enforce_linear_iteration_=*/false);
    ti_build::allocate_or_resize_outputs(shape, perm, operands, /*num_outputs_=*/1, has_coalesced_dimensions, set_output);
    ti_build::coalesce_dimensions(shape, operands, has_coalesced_dimensions);
  }

  Layout r{to_ints<T>(shape), {}, operands[0].tensor.sizes().vec(), operands[0].tensor.strides().vec()};
  for (const auto& op : operands) {
    r.strides.push_back(to_ints<T>(op.stride_bytes));
  }
  return r;
}

Layout from_iterator(const Tensor& a, const Tensor& b) {
  Tensor out;
  auto iter = TensorIterator::binary_op(out, a, b);
  Layout r{iter.shape().vec(), {}, iter.output().sizes().vec(), iter.output().strides().vec()};
  for (const auto i : c10::irange(iter.ntensors())) {
    r.strides.push_back(iter.strides(i).vec());
  }
  return r;
}

void expect_same(const Layout& x, const Layout& y) {
  EXPECT_EQ(x.shape, y.shape);
  EXPECT_EQ(x.strides, y.strides);
  EXPECT_EQ(x.out_sizes, y.out_sizes);
  EXPECT_EQ(x.out_strides, y.out_strides);
}

} // namespace

// The c10::SymInt instantiation of the TensorIterator build, on concrete
// SymInts, matches the int64_t one and TensorIterator itself.
TEST(TensorIteratorBuildTest, SymIntMatchesInt64) {
  std::vector<std::pair<Tensor, Tensor>> cases = {
      {at::randn({8, 16}), at::randn({8, 16})},
      {at::randn({8, 16}), at::randn({16})},
      {at::randn({8, 16}), at::randn({8, 1})},
      {at::randn({16, 8}).t(), at::randn({8, 16})},
      {at::randn({4, 8, 16}), at::randn({16, 8, 4}).permute({2, 1, 0})},
      {at::randn({2, 3, 4, 5}).contiguous(MemoryFormat::ChannelsLast), at::randn({2, 3, 4, 5}).contiguous(MemoryFormat::ChannelsLast)},
      {at::randn({2, 3, 4, 5}).permute({0, 2, 1, 3}), at::randn({2, 4, 3, 5})},
      {at::randn({1, 7}), at::randn({5, 1})},
      {at::randn({}), at::randn({3, 4})},
      {at::randn({0, 4}), at::randn({4})},
  };
  for (const auto& [a, b] : cases) {
    auto ref = from_iterator(a, b);
    expect_same(build<int64_t>({a, b}), ref);
    expect_same(build<c10::SymInt>({a, b}), ref);
  }
}

// TensorIteratorSym on concrete SymInts: the output layout after build and the
// coalesced shape and strides match TensorIterator.
TEST(TensorIteratorBuildTest, TensorIteratorSymMatchesTensorIterator) {
  std::vector<std::pair<Tensor, Tensor>> cases = {
      {at::randn({8, 16}), at::randn({16})},
      {at::randn({16, 8}).t(), at::randn({8, 16})},
      {at::randn({4, 8, 16}), at::randn({16, 8, 4}).permute({2, 1, 0})},
      {at::randn({2, 3, 4, 5}).contiguous(MemoryFormat::ChannelsLast), at::randn({2, 3, 4, 5}).contiguous(MemoryFormat::ChannelsLast)},
      {at::randn({1, 7}), at::randn({5, 1})},
  };
  for (const auto& [a, b] : cases) {
    TensorIteratorSym iter(TensorIteratorConfig().add_owned_output(TensorBase()).add_const_input(a).add_const_input(b));
    auto ref = from_iterator(a, b);
    EXPECT_EQ(iter.output().sizes().vec(), ref.out_sizes);
    EXPECT_EQ(iter.output().strides().vec(), ref.out_strides);
    iter.coalesce_dimensions();
    EXPECT_EQ(to_ints<c10::SymInt>(iter.shape()), ref.shape);
    for (const auto i : c10::irange(iter.ntensors())) {
      EXPECT_EQ(to_ints<c10::SymInt>(iter.strides(i)), ref.strides[i]);
    }
  }
}
