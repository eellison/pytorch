#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/script/schema_matching.h>

namespace torch {
namespace jit {
namespace script {

using SugaredValuePtr = std::shared_ptr<SugaredValue>;

// The AST can contain nodes like `self`, `self.b` or `python_fn` that
// are not first-class values in the graph representation, but instead
// will be desugared based on how they are used in the AST.

// SugaredValue is used to temporarily represent these values in a way
// that separates their behavior from the AST -> IR converter itself.
// This allows us to keep dependencies on python minimal.

struct IterableValue;
using IterableValuePtr = std::shared_ptr<IterableValue>;

struct TORCH_API SugaredValue
    : public std::enable_shared_from_this<SugaredValue> {
  // what is this node? for error reporting (e.g. Module, python function)
  virtual std::string kind() const = 0;

  // what can we do with this thing?
  // use it as a value e.g.  `this + 4`
  virtual Value* asValue(const SourceRange& loc, Function& m) {
    throw ErrorReport(loc) << kind() << " cannot be used as a value";
  }

  // select an attribute on it, e.g. `this.field`
  virtual std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) {
    throw ErrorReport(loc) << "attribute lookup is not defined on " << kind();
  }

  // assign an attribute on it, e.g. `this.field = newValue`
  virtual void setAttr(
      const SourceRange& loc,
      Function& m,
      const std::string& field,
      Value* newValue) {
    throw ErrorReport(loc) << "attribute assignment is not defined on "
                           << kind();
  }

  // use it as a vector of values, e.g. a tuple of values as return value from
  // a method invocation
  virtual std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      Function& m,
      const c10::optional<size_t>& size_hint = {}) {
    throw ErrorReport(loc) << kind() << " cannot be used as a tuple";
  }

  virtual std::vector<std::shared_ptr<SugaredValue>> asType(
      const SourceRange& loc,
      Method& m) {
    throw ErrorReport(loc) << kind() << " cannot be used as a type";
  }

  virtual IterableValuePtr asIterable(
      const SourceRange& loc,
      Function& m) {
    throw ErrorReport(loc) << kind() << " cannot be used as an iterable";
  }

  // call it like a function, e.g. `outputs = this(inputs)`
  virtual std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      // note: names for args will be 'argument 0', 'argument 1', etc..
      at::ArrayRef<NamedValue> inputs_,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) {
    // n_binders is always set to the number of variables an expression is
    // syntactically bound to:
    //     a = foo() # 1 binder (note in this case the single binder might be a
    //     tuple) a, * b = foo() # 1 binder a, b = foo() # 2 binders foo() # 0
    //     binders
    //
    // In subexpressions, like bar() in foo(bar()), n_binders is always set to
    // 1. n_binders is used as a hint to subexpressions to determine how many
    // values they should return when that number is ambiguous statically. In
    // particular it is currently used to decide how many tensors a call to a
    // python function will return. It is only a hint, functions do not have to
    // check that n_binders match the number of things they are returning, the
    // assignment logic will do that anyway.

    throw ErrorReport(loc) << "cannot call a " << kind();
  }

  // return length of this thing, if not then it can't be iterated.
  virtual Value* len(const SourceRange& loc, Function& m) {
    throw ErrorReport(loc) << "'" << kind() << "'"
                           << " object is not iterable";
  }
  // expression for ith elemement for iterable value
  virtual SugaredValuePtr getitem(const SourceRange& loc, Function& m, Value* idx) {
    throw ErrorReport(loc) << "'" << kind() << "'"
                           << " object is not subscriptable";
  }

  virtual ~SugaredValue() = default;
};

// most things in the environment are just simple value types
// and not special python syntax sugar types
struct TORCH_API SimpleValue : public SugaredValue {
  SimpleValue(Value* value) : value_(value) {}
  std::string kind() const override {
    std::stringstream ss;
    ss << "value of type '" << value_->type()->python_str() << "'";
    return ss.str();
  }
  Value* asValue(const SourceRange& range, Function& m) override {
    return value_;
  }
  std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      Function& m,
      const c10::optional<size_t>& size_hint = {}) override;
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override;

  void setAttr(
      const SourceRange& loc,
      Function& m,
      const std::string& field,
      Value* newValue) override;

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      // note: names for args will be 'argument 0', 'argument 1', etc..
      at::ArrayRef<NamedValue> inputs_,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override;

  IterableValuePtr asIterable(const SourceRange& loc, Function& m) override;

  Value* getValue() const {
    return value_;
  }

  Value* len(const SourceRange& loc, Function& m) override;
  SugaredValuePtr getitem(const SourceRange& loc, Function& m, Value* idx) override;

 private:
  Value* value_;
};

struct TORCH_API BuiltinFunction : public SugaredValue {
  BuiltinFunction(Symbol symbol, c10::optional<NamedValue> self)
      : symbol(symbol), self(std::move(self)) {}

  // The symbol of the function (e.g. `aten::relu`).
  Symbol symbol;

  // if this is method, then this is the self argument.
  c10::optional<NamedValue> self;

  std::string kind() const override {
    return "builtin";
  }
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> attributes,
      at::ArrayRef<NamedValue> inputs,
      size_t n_binders) override;

  // try to create this builtin but if it doesn't exist or the self argument
  // cannot possibly match, then return nullptr. Use in situations where it is
  // not clear if it is a valid builtin
  static std::shared_ptr<BuiltinFunction> tryCreate(
      Symbol symbol,
      c10::optional<NamedValue> self);
};

struct TORCH_API SugaredTupleValue : public SugaredValue {
  explicit SugaredTupleValue(
      std::vector<std::shared_ptr<SugaredValue>> tup, bool emit_unrolled)
      : tup_(tup), emit_unrolled_(emit_unrolled) {};

  std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      Function& m,
      const c10::optional<size_t>& size_hint = {}) override {
    return tup_;
  };

  Value * asValue(const SourceRange &loc, Function &m) override {
    std::vector<Value*> vec;
    for (const auto& sv: tup_) {
      vec.push_back(sv->asValue(loc, m));
    }
    Graph& g = *m.graph();
    return g.insertNode(g.createTuple(vec))->output();
  }

  std::string kind() const override {
    return "Sugared Tuple";
  }

  SugaredValuePtr getitem(const SourceRange& loc, Function& m, Value* idx) override {
    TORCH_INTERNAL_ASSERT(toIValue(idx), loc, "Expected integer literal for Sugared Tuple");
    auto index = toIValue(idx)->toInt();
    TORCH_INTERNAL_ASSERT(index >= 0 && index < static_cast<int64_t>(tup_.size()),
      loc,
      "Index out of range of Sugared Tuple");
    return tup_.at(index);
  }

  IterableValuePtr asIterable(const SourceRange& loc, Function& m) override {
    return std::make_shared<IterableValue>(std::make_shared<SugaredTupleValue>(tup_, emit_unrolled_), tup_.size(), emit_unrolled_);
  };


  std::vector<std::shared_ptr<SugaredValue>> tup_;
  bool emit_unrolled_;
};


struct TORCH_API BuiltinModule : public SugaredValue {
  BuiltinModule(std::string name, c10::optional<int64_t> version = at::nullopt)
      : name(std::move(name)), version(std::move(version)) {}

  std::string kind() const override {
    return "builtin module";
  }
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override {
    if (field == "autograd") {
      // When refering torch.autograd, it is also considered to be a
      // BuiltinModule and we will dispatch to the aten operators for the
      // methods under its module.
      return std::make_shared<BuiltinModule>("aten", version);
    }
    return std::make_shared<BuiltinFunction>(
        Symbol::fromQualString(name + "::" + field), c10::nullopt);
  }

 private:
  std::string name;
  // when we add operator versioning, emit this op as it exising at 'version'
  // if not set, use the latest version
  c10::optional<int64_t> version;
};

// Represents a class, analagous to `int` or `dict`. Instances of classes,
// like `1` or `{"foo": 5}`, are represented as SimpleValues
struct TORCH_API ClassValue : public SugaredValue {
  explicit ClassValue(ClassTypePtr type) : type_(std::move(type)) {}

  // Call the type's constructor, as in:
  //    n = Foo(constructor_arg)
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override;

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override;

  std::string kind() const override {
    return type_->str();
  }

  ClassTypePtr type_;
};

struct TORCH_API NamedTupleConstructor : public SugaredValue {
  explicit NamedTupleConstructor(TupleTypePtr type) : type_(std::move(type)) {}

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override;

  std::string kind() const override {
    return type_->str();
  }

  TupleTypePtr type_;
};

struct FunctionValue : public SugaredValue {
  FunctionValue(Function* callee) : callee_(std::move(callee)) {}
  FunctionValue(const StrongFunctionPtr& p)
      : callee_(p.function_), cu_(p.cu_) {}

  std::string kind() const override {
    return "function";
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& f,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
    callee_->ensure_defined();
    MatchedSchema match =
        matchSchema(callee_->getSchema(), loc, *f.graph(), inputs, attributes);
    Value* output = f.graph()->insertFunctionCall(callee_, match);
    output->node()->setSourceRange(loc);
    return std::make_shared<SimpleValue>(output);
  }

 private:
  Function* callee_;
  // TODO holding this thing is creepy
  std::shared_ptr<CompilationUnit> cu_;
};

struct TORCH_API ClosureValue : public SugaredValue {
  ClosureValue(Value* value) : value_(value) {
    TORCH_INTERNAL_ASSERT(value_->node()->kind() == prim::Function);
  }
  std::string kind() const override {
    return "closure";
  }
  Value* asValue(const SourceRange& range, Function& m) override {
    return value_;
  }
  Value* value_;
};

// defines how a method obtained from a module/class/interface behaves in script
struct MethodValue : public SugaredValue {
  MethodValue(Value* self, std::string method_name)
      : self_(std::move(self)), method_name_(std::move(method_name)) {}

  std::string kind() const override {
    return "method";
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& f,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
    std::vector<NamedValue> inputsWithSelf = {self_};
    inputsWithSelf.insert(inputsWithSelf.end(), inputs.begin(), inputs.end());
    const FunctionSchema* schema = nullptr;
    if (auto class_type = self_->type()->cast<ClassType>()) {
      auto method = class_type->getMethod(method_name_);
      TORCH_INTERNAL_ASSERT(method);
      method->ensure_defined();
      schema = &method->getSchema();
    } else if (auto interface_type = self_->type()->cast<InterfaceType>()) {
      schema = interface_type->getMethod(method_name_);
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "method constructed that is not a class or interface");
    }
    MatchedSchema match =
        matchSchema(*schema, loc, *f.graph(), inputsWithSelf, attributes);
    Value* output = f.graph()->insertMethodCall(method_name_, match);
    output->node()->setSourceRange(loc);
    return std::make_shared<SimpleValue>(output);
  }

 private:
  Value* self_;
  std::string method_name_;
};

struct TORCH_API PrintValue : public SugaredValue {
  std::string kind() const override {
    return "print";
  }
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override;
};

// expressions like int(x)
// these are the same as call prim::Int or equivalent except it
// is a noop when the input is a subtype of 'type'
struct TORCH_API CastValue : public BuiltinFunction {
  CastValue(TypePtr type, c10::Symbol method)
      : BuiltinFunction(method, c10::nullopt), type_(std::move(type)) {}
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
    if (inputs.size() == 1 && attributes.size() == 0) {
      auto v = inputs[0].value(*m.graph());
      if (v->type()->isSubtypeOf(type_)) {
        return std::make_shared<SimpleValue>(v);
      }
    }
    return BuiltinFunction::call(loc, m, inputs, attributes, n_binders);
  }

 private:
  TypePtr type_;
};


// builtins operators and functions that call a method if it exists
// on a class type, like 'len(x)' and 'x + y'
struct TORCH_API MagicMethod : public SugaredValue {
  MagicMethod(std::string desugared_name, SugaredValuePtr base)
      : base_value_(std::move(base)),
        desugared_name_(std::move(desugared_name)) {}

  std::string kind() const override {
    return desugared_name_;
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override;

 private:
  SugaredValuePtr base_value_;
  std::string desugared_name_;
};

// things that look like function applications, but
// perform non-standard evaluation are represented
// with SpecialFormValues, e.g.
//   isinstance(x, int)
//   fork(fn)
//   annotate(int, 3)
// The implementation of each value is handled by a case inside emitApplyExpr
struct TORCH_API SpecialFormValue : public SugaredValue {
  SpecialFormValue(Symbol form) : form_(form) {}
  std::string kind() const override {
    return form_.toUnqualString();
  }
  Symbol form() const {
    return form_;
  }
  static std::shared_ptr<SpecialFormValue> create(Symbol form) {
    return std::make_shared<SpecialFormValue>(form);
  }

 private:
  Symbol form_;
};


// matched against for special handling of range expressions
struct TORCH_API RangeValue : SugaredValue {
  RangeValue(const SourceRange& loc, Function& m, std::vector<Value*> inputs);

  std::string kind() const override {
    return "range";
  }
  Value* len(const SourceRange& loc, Function& m) override;
  c10::optional<int64_t> staticLen();

  SugaredValuePtr getitem(const SourceRange& loc, Function& m, Value* idx) override;

  IterableValuePtr asIterable(const SourceRange& loc, Function& m) override;

 private:
  Value* start_;
  Value* end_;
  Value* step_;
  // a flag to determine if it's a simple range() call with only end_ from
  // arguments If true, we will not insert length calculation and index
  // derivation nodes to simplify the graph and enable more possible
  // optimizations
  bool has_only_end_;
  c10::optional<int64_t> static_len_;
};


// We handle iteration over Module Containers by unrolling the for loop over each value.
// As a result we need to statically know the number of elements of the iterable.
// IterableValue contains an underlying SugaredValue, its static length if it is known,
// and whether or not the Iterable needs to be emmitted statically.
// We error if an iterable contains both a SugaredValue that needs to be emitted statically,
// and a SugaredValue which does not have a statically-determinable length.
struct IterableValue {
  IterableValue(
      SugaredValuePtr value,
      c10::optional<int64_t> len = c10::nullopt,
      bool emit_unrolled = false)
      : value_(std::move(value)), len_(len), emit_unrolled_(emit_unrolled){};

  SugaredValuePtr getValue() const {
    return value_;
  }

  c10::optional<int64_t> getLen() const {
    return len_;
  }

  bool emitUnrolled() const {
    return emit_unrolled_;
  }

private:
  std::shared_ptr<SugaredValue> value_;
  c10::optional<int64_t> len_;
  bool emit_unrolled_ = false;
};


// Specialized Tree structure to matched against for special handling
// of builtin functions iterables expressions like zip(), enumerate(), etc.
// zip and enumerate can be modeled as a tree of SimpleValue/RangeValue:
//    zip(x, y) ->  (x, y) with tuple assignment to each loop target
//    enumerate(x) -> (range(0, math.inf, 1), x)
// So a complicated expression like zip(a, enumerate(b), range(0, 100)) will be:
// (a, (range(0, math.inf, 1), b), range(0, 100))
// We use those base iterables to fill in the loop information like
// max_trip_count and set the value table for loop targets
struct TORCH_API IterableTree : SugaredValue {
  IterableTree() = default;
  IterableTree(const SourceRange& range, at::ArrayRef<IterableValuePtr> children) {
    for (const auto& child: children) {
      addChild(range, child);
    }
  }
  std::string kind() const override {
    return "iterabletree";
  }

  IterableValuePtr asIterable(const SourceRange& loc, Function& m) override {
    return std::make_shared<IterableValue>(shared_from_this(), static_len_, emit_unrolled_);
  }

  void addChild(const SourceRange& range, IterableValuePtr iter_value) {
    auto child_len = iter_value->getLen();
    auto child_unrolled = iter_value->emitUnrolled();
    if (children_.size() == 0) {
      static_len_ = child_len;
      emit_unrolled_ = child_unrolled;
    } else {
      if ((emit_unrolled_ && !child_len) ||
          (child_unrolled && !static_len_)) {
        throw ErrorReport(range)
            << "Can not iterate over a module list with a value "
               "that does not have a statically determinable length\n";
      }
      if (child_len && static_len_) {
        // iterables run for the minimum length of all its leaves
        static_len_ = std::min(*child_len, *static_len_);
      }
      emit_unrolled_ = emit_unrolled_ || child_unrolled;
    }

    children_.push_back(iter_value->getValue());
  }

  std::vector<SugaredValuePtr> get_children() {
    return children_;
  }

  c10::optional<int64_t> staticLen() const {
    return static_len_;
  }

  bool emitUnrolled() const {
    return emit_unrolled_;
  }

  // given a IterableTree node, get all the base iterables/leaves under the
  // IterableTree node. This enables
  // us to get all the basic SugaredValues that contains valid loop information
  // with len() and getitem()
  std::vector<SugaredValuePtr> get_base_iterables();

  Value* len(const SourceRange& loc, Function& m) override;
  SugaredValuePtr getitem(const SourceRange& loc, Function& m, Value* idx) override;

 private:
  c10::optional<int64_t> static_len_ = c10::nullopt;
  bool emit_unrolled_ = false;
  std::vector<SugaredValuePtr> children_;
};


static inline std::vector<Value*> toValues(
    Graph& g,
    at::ArrayRef<NamedValue> nvs) {
  return fmap(nvs, [&](const NamedValue& v) { return v.value(g); });
}

struct SimpleSelf : public Self {
  explicit SimpleSelf(ClassTypePtr classType)
      : Self(), classType_(std::move(classType)) {}
  std::shared_ptr<SugaredValue> makeSugared(Value* v) const override {
    v->setType(classType_);
    return std::make_shared<SimpleValue>(v);
  }
  ClassTypePtr getClassType() const override {
    return classType_;
  }

 private:
  ClassTypePtr classType_;
};
} // namespace script
} // namespace jit
} // namespace torch
