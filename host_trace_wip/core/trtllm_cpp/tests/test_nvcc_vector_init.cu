// nvcc (cudafe) front-end bug, CUDA 13.0: inside a template, `auto s = std::vector<c10::SymInt>{a, b, c}` has one element
// (f3, f4 print 1); g++ prints 2 2 2 3 2. Why ht/make_patch.py parenthesizes the TMA shape lists.
//   nvcc -std=c++20 test_nvcc_vector_init.cu -I$TORCH/include -L$TORCH/lib -lc10 && ./a.out   # prints 2 2 1 1 2
#include <c10/core/SymInt.h>
#include <vector>
#include <cstdio>
struct O { int a = 128; int b = 4; };
template <class X> static auto f1(X const& o) { return std::vector<c10::SymInt>({static_cast<c10::SymInt>(o.a), static_cast<c10::SymInt>(o.b)}); }
template <class X> static auto f2(X const& o) { std::vector<c10::SymInt> s{static_cast<c10::SymInt>(o.a), static_cast<c10::SymInt>(o.b)}; return s; }
template <class X> static auto f3(X const& o) { auto s = std::vector<c10::SymInt>{1, static_cast<c10::SymInt>(o.b)}; return s; }
template <class X> static auto f4(X const& o) { auto s = std::vector<c10::SymInt>{static_cast<c10::SymInt>(o.a), static_cast<c10::SymInt>(o.b), static_cast<c10::SymInt>(o.b)}; return s; }
template <class X> static auto f5(X const& o) { auto s = std::vector<int64_t>{static_cast<int64_t>(o.a), static_cast<int64_t>(o.b)}; return s; }
int main() { O o; printf("%zu %zu %zu %zu %zu\n", f1(o).size(), f2(o).size(), f3(o).size(), f4(o).size(), f5(o).size()); }
