#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace parameter_program {

enum class Op : uint32_t {
  Constant, Value, Pointer, Add, Sub, Mul, UDiv, SDiv, URem, SRem,
  And, Or, Xor, Shl, LShr, AShr, Trunc, ZExt, SExt, ICmp, Select
};

enum class Predicate : uint32_t { Eq, Ne, Ult, Ule, Ugt, Uge, Slt, Sle, Sgt, Sge };
enum Flag : uint32_t { NUW = 1, NSW = 2, Exact = 4 };

struct Instruction {
  Op op;
  uint32_t width;
  uint32_t first = 0;
  uint32_t second = 0;
  uint32_t third = 0;
  uint32_t flags = 0;
  Predicate predicate = Predicate::Eq;
  uint64_t immediate = 0;
};

struct Value {
  uint64_t bits = 0;
  bool poison = false;
};

class Program {
 public:
  Program(std::vector<Instruction> instructions, std::vector<uint32_t> outputs,
          size_t early_count, size_t root_count)
      : instructions_(std::move(instructions)), outputs_(std::move(outputs)),
        early_count_(early_count), root_count_(root_count) {
    static_assert(sizeof(uintptr_t) == sizeof(uint64_t));
    std::vector<std::vector<size_t>> roots(instructions_.size());
    for (size_t index = 0; index < instructions_.size(); ++index) {
      const auto& row = instructions_[index];
      require(row.width == 1 || row.width == 32 || row.width == 64, "Unsupported integer width");
      uint32_t allowed_flags = 0;
      if (row.op == Op::Add || row.op == Op::Sub || row.op == Op::Mul || row.op == Op::Shl) {
        allowed_flags = NUW | NSW;
      } else if (row.op == Op::UDiv || row.op == Op::SDiv || row.op == Op::LShr || row.op == Op::AShr) {
        allowed_flags = Exact;
      }
      require((row.flags & ~allowed_flags) == 0, "Unsupported instruction flags");
      auto operand = [&](uint32_t source) -> uint32_t {
        require(source < index, "SSA operands must precede their result");
        roots[index].insert(roots[index].end(), roots[source].begin(), roots[source].end());
        return instructions_[source].width;
      };
      switch (row.op) {
        case Op::Constant:
          require((row.immediate & ~mask(row.width)) == 0, "Constant exceeds its integer width");
          break;
        case Op::Value:
          require(row.width != 1 && row.first < early_count_, "Invalid early value input");
          break;
        case Op::Pointer:
          require(row.width == 64 && row.first < root_count_ && row.second < early_count_,
                  "Invalid pointer root or byte displacement");
          roots[index].push_back(row.first);
          pointer_inputs_.push_back(row.first);
          break;
        case Op::Trunc:
          require(operand(row.first) > row.width, "Trunc must narrow its operand");
          break;
        case Op::ZExt:
        case Op::SExt:
          require(operand(row.first) < row.width, "Extension must widen its operand");
          break;
        case Op::ICmp: {
          const auto width = operand(row.first);
          require(row.width == 1 && operand(row.second) == width &&
                      static_cast<uint32_t>(row.predicate) <= static_cast<uint32_t>(Predicate::Sge),
                  "Invalid integer comparison");
          break;
        }
        case Op::Select:
          require(operand(row.first) == 1 && operand(row.second) == row.width &&
                      operand(row.third) == row.width, "Invalid select operand types");
          break;
        case Op::And:
        case Op::Or:
        case Op::Xor:
          require(operand(row.first) == row.width && operand(row.second) == row.width,
                  "Bitwise operand types differ");
          break;
        case Op::Add:
        case Op::Sub:
        case Op::Mul:
        case Op::UDiv:
        case Op::SDiv:
        case Op::URem:
        case Op::SRem:
        case Op::Shl:
        case Op::LShr:
        case Op::AShr:
          require(row.width != 1 && operand(row.first) == row.width && operand(row.second) == row.width,
                  "Arithmetic operand types differ");
          break;
        default:
          throw std::invalid_argument("Unsupported parameter instruction");
      }
      auto& used = roots[index];
      std::sort(used.begin(), used.end());
      used.erase(std::unique(used.begin(), used.end()), used.end());
    }
    for (auto output : outputs_) {
      require(output < instructions_.size() && instructions_[output].width != 1,
              "Physical outputs must name i32 or i64 instructions");
      output_roots_.push_back(roots[output]);
    }
  }

  size_t scratch_size() const { return instructions_.size(); }
  size_t output_count() const { return outputs_.size(); }
  uint32_t output_width(size_t index) const { return instructions_[outputs_[index]].width; }
  const std::vector<size_t>& pointer_inputs() const { return pointer_inputs_; }
  const std::vector<std::vector<size_t>>& output_roots() const { return output_roots_; }

  void evaluate(const int64_t* early, const uintptr_t* pointers, Value* scratch, int64_t* outputs) const {
    for (size_t index = 0; index < instructions_.size(); ++index) {
      const auto& row = instructions_[index];
      const auto width = row.width;
      const auto limit = mask(width);
      Value result;
      if (row.op == Op::Constant) {
        result.bits = row.immediate;
      } else if (row.op == Op::Value) {
        require(width != 32 || (early[row.first] >= std::numeric_limits<int32_t>::min() &&
                                   early[row.first] <= std::numeric_limits<uint32_t>::max()),
                "Early value cannot represent an i32 input without truncation");
        result.bits = static_cast<uint64_t>(early[row.first]) & limit;
      } else if (row.op == Op::Pointer) {
        const auto base = pointers[row.first];
        const auto offset = early[row.second];
        if (offset >= 0) {
          const auto increment = static_cast<uint64_t>(offset);
          require(increment <= std::numeric_limits<uintptr_t>::max() - base, "Pointer addition overflow");
          result.bits = base + increment;
        } else {
          const auto decrement = uint64_t{0} - static_cast<uint64_t>(offset);
          require(decrement <= base, "Pointer subtraction underflow");
          result.bits = base - decrement;
        }
      } else if (row.op == Op::Select) {
        const auto condition = scratch[row.first];
        result = condition.poison ? Value{0, true} : scratch[condition.bits ? row.second : row.third];
      } else {
        const auto left = scratch[row.first];
        const auto source_width = instructions_[row.first].width;
        if (row.op == Op::Trunc || row.op == Op::ZExt || row.op == Op::SExt) {
          result = left;
          if (row.op == Op::SExt && (left.bits & sign_bit(source_width))) {
            result.bits |= ~mask(source_width);
          }
        } else {
          const auto right = scratch[row.second];
          result.poison = left.poison || right.poison;
          if (!result.poison) {
            const auto a = left.bits;
            const auto b = right.bits;
            const auto sa = signed_value(a, source_width);
            const auto sb = signed_value(b, source_width);
            switch (row.op) {
              case Op::Add:
                result.bits = a + b;
                result.poison = overflow(row, Wide(a) + b, sa + sb);
                break;
              case Op::Sub:
                result.bits = a - b;
                result.poison = ((row.flags & NUW) && a < b) || signed_overflow(row, sa - sb);
                break;
              case Op::Mul:
                result.bits = a * b;
                result.poison = overflow(row, Wide(a) * b, sa * sb);
                break;
              case Op::UDiv:
              case Op::URem:
                require(b != 0, "Unsigned division by zero is undefined");
                result.bits = row.op == Op::UDiv ? a / b : a % b;
                result.poison = (row.flags & Exact) && a % b != 0;
                break;
              case Op::SDiv:
              case Op::SRem:
                require(b != 0, "Signed division by zero is undefined");
                require(sa != -(SignedWide(1) << (width - 1)) || sb != -1,
                        "Signed division overflow is undefined");
                result.bits = static_cast<uint64_t>(row.op == Op::SDiv ? sa / sb : sa % sb);
                result.poison = (row.flags & Exact) && sa % sb != 0;
                break;
              case Op::And: result.bits = a & b; break;
              case Op::Or: result.bits = a | b; break;
              case Op::Xor: result.bits = a ^ b; break;
              case Op::Shl:
                result.poison = b >= width;
                if (!result.poison) {
                  result.bits = a << b;
                  result.poison = overflow(row, Wide(a) << b, sa * (SignedWide(1) << b));
                }
                break;
              case Op::LShr:
              case Op::AShr:
                result.poison = b >= width;
                if (!result.poison) {
                  result.bits = a >> b;
                  if (row.op == Op::AShr && b && (a & sign_bit(width))) {
                    result.bits |= limit ^ (limit >> b);
                  }
                  result.poison = (row.flags & Exact) && b && (a & ((uint64_t{1} << b) - 1));
                }
                break;
              case Op::ICmp:
                switch (row.predicate) {
                  case Predicate::Eq: result.bits = a == b; break;
                  case Predicate::Ne: result.bits = a != b; break;
                  case Predicate::Ult: result.bits = a < b; break;
                  case Predicate::Ule: result.bits = a <= b; break;
                  case Predicate::Ugt: result.bits = a > b; break;
                  case Predicate::Uge: result.bits = a >= b; break;
                  case Predicate::Slt: result.bits = sa < sb; break;
                  case Predicate::Sle: result.bits = sa <= sb; break;
                  case Predicate::Sgt: result.bits = sa > sb; break;
                  case Predicate::Sge: result.bits = sa >= sb; break;
                }
                break;
              default: throw std::logic_error("Unvalidated parameter instruction");
            }
          }
        }
      }
      result.bits &= limit;
      scratch[index] = result;
    }
    for (auto output : outputs_) {
      require(!scratch[output].poison, "Live poison cannot initialize a physical parameter field");
    }
    for (size_t index = 0; index < outputs_.size(); ++index) {
      const auto output = outputs_[index];
      auto bits = scratch[output].bits;
      if (instructions_[output].width == 32 && (bits & sign_bit(32))) {
        bits |= ~mask(32);
      }
      std::memcpy(outputs + index, &bits, sizeof(bits));
    }
  }

 private:
  using Wide = unsigned __int128;
  using SignedWide = __int128;

  static void require(bool condition, const char* message) {
    if (!condition) {
      throw std::invalid_argument(message);
    }
  }

  static uint64_t mask(uint32_t width) {
    return width == 64 ? std::numeric_limits<uint64_t>::max() : (uint64_t{1} << width) - 1;
  }

  static uint64_t sign_bit(uint32_t width) { return uint64_t{1} << (width - 1); }

  static SignedWide signed_value(uint64_t bits, uint32_t width) {
    return (bits & sign_bit(width)) ? SignedWide(bits) - (SignedWide(1) << width) : SignedWide(bits);
  }

  static bool signed_overflow(const Instruction& row, SignedWide result) {
    const auto bound = SignedWide(1) << (row.width - 1);
    return (row.flags & NSW) && (result < -bound || result >= bound);
  }

  static bool overflow(const Instruction& row, Wide unsigned_result, SignedWide signed_result) {
    return ((row.flags & NUW) && unsigned_result > mask(row.width)) || signed_overflow(row, signed_result);
  }

  std::vector<Instruction> instructions_;
  std::vector<uint32_t> outputs_;
  size_t early_count_;
  size_t root_count_;
  std::vector<std::vector<size_t>> output_roots_;
  std::vector<size_t> pointer_inputs_;
};

} // namespace parameter_program
