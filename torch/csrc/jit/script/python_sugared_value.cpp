#include <torch/csrc/Dtype.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/jit/script/module_python.h>
#include <torch/csrc/jit/script/python_sugared_value.h>
#include <torch/csrc/jit/script/schema_matching.h>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <Python.h>

namespace torch {
namespace jit {
namespace script {

std::string typeString(py::handle h) {
  return py::str(h.get_type().attr("__name__"));
}

c10::optional<StrongFunctionPtr> as_function(const py::object& obj) {
  if (py::isinstance<StrongFunctionPtr>(obj)) {
    return py::cast<StrongFunctionPtr>(obj);
  }
  return c10::nullopt;
}

FunctionSchema PythonValue::getSchema(
    const size_t n_args,
    const size_t n_binders,
    const SourceRange& loc) {
  auto annotations = py::module::import("torch.jit.annotations");
  const auto fn_to_get_signature =
      moduleSelf_ ? py::getattr(self, "original_fn") : self;
  auto signature = annotations.attr("get_signature")(
      fn_to_get_signature, rcb ? *rcb : py::none(), loc);
  std::vector<Argument> args, rets;

  if (moduleSelf_) {
    args.push_back(Argument("self", moduleSelf_->type(), {}, {}, false));
  }
  // We may mutate this if we can determine the number of args from Python
  // introspection.
  size_t actual_n_args = moduleSelf_ ? n_args + 1 : n_args;
  if (!signature.is_none()) {
    std::vector<TypePtr> arg_types;
    TypePtr ret_type;
    std::tie(arg_types, ret_type) =
        py::cast<std::pair<std::vector<TypePtr>, TypePtr>>(signature);
    args.reserve(arg_types.size());
    size_t idx = 0; // Fake argument names by putting in the index
    for (auto& arg_type : arg_types) {
      args.push_back(
          Argument(std::to_string(idx++), std::move(arg_type), {}, {}, false));
    }
    rets.push_back(Argument("0", std::move(ret_type), {}, {}, false));
  } else {
    // Create a default signature using what information we have

    // First see if we can introspect the number of function parameters
    // irrespective of the presence of explicit type annotations
    auto num_params =
        annotations.attr("get_num_params")(fn_to_get_signature, loc);
    if (!num_params.is_none()) {
      // Return a signature with the correct number of params according to the
      // Python function. The error handling in call() will catch any mismatch
      // later.
      actual_n_args = py::cast<size_t>(num_params);
      if (moduleSelf_) {
        TORCH_INTERNAL_ASSERT(actual_n_args > 0);
        --actual_n_args;
      }
    }
    // Construct the default signature: all arguments and returns will be
    // DynamicType
    args.reserve(actual_n_args);
    for (size_t i = 0; i < actual_n_args; ++i) {
      args.push_back(
          Argument(std::to_string(i), TensorType::get(), {}, {}, false));
    }
    TypePtr ret_type = TensorType::get();
    if (n_binders == 0) {
      ret_type = NoneType::get();
    } else if (n_binders > 1) {
      std::vector<TypePtr> tuple_values(n_binders, ret_type);
      ret_type = TupleType::create(std::move(tuple_values));
    }
    rets.push_back(Argument("0", ret_type, {}, {}, false));
  }
  std::string name("");
  // Use the qualified name if possible
  if (py::hasattr(self, "__qualname__")) {
    name = py::str(py::getattr(self, "__qualname__"));
  } else if (py::hasattr(self, "__name__")) {
    name = py::str(py::getattr(self, "__name__"));
  }
  return FunctionSchema("", "", std::move(args), std::move(rets));
}

std::shared_ptr<SugaredValue> PythonValue::call(
    const SourceRange& loc,
    Function& m,
    at::ArrayRef<NamedValue> inputs_,
    at::ArrayRef<NamedValue> attributes,
    size_t n_binders) {
  std::vector<NamedValue> inputsWithSelf;
  if (moduleSelf_) {
    inputsWithSelf.emplace_back(NamedValue("self", moduleSelf_));
  }
  inputsWithSelf.insert(inputsWithSelf.end(), inputs_.begin(), inputs_.end());
  inputs_ = inputsWithSelf;

  auto schema = getSchema(inputs_.size(), n_binders, loc);
  auto inputs = toValues(*m.graph(), inputs_);

  std::stringstream failure_messages;
  c10::optional<MatchedSchema> matched_schema = tryMatchSchema(
      schema,
      loc,
      *m.graph(),
      c10::nullopt,
      inputs_,
      attributes,
      &failure_messages,
      /*conv_tensor_to_num*/ true);
  if (!matched_schema)
    throw ErrorReport(loc) << failure_messages.str();

  // If if a function is marked as dropped,
  // we throw an exception if it is invoked.
  if (py::cast<bool>(py::module::import("torch._jit_internal")
                         .attr("should_drop")(self))) {
    auto g = m.graph();
    auto err_msg = insertConstant(
        *g,
        IValue(
            "This Python function is annotated to be ignored and cannot be run"));
    g->insert(prim::RaiseException, {err_msg}, {}, loc);
    return std::make_shared<SimpleValue>(
        g->insertNode(
             g->createUninitialized(matched_schema->return_types.at(0)))
            ->output());
  }

  // Release the function object so we can wrap it in a PythonOp
  py::object func = self;
  std::string cconv(inputs.size(), 'd');
  Node* new_node = m.graph()->insertNode(
      m.graph()->createPythonOp(THPObjectPtr(func.release().ptr()), cconv, {}));

  new_node->setSourceRange(loc);
  for (auto& i : matched_schema->inputs)
    new_node->addInput(i);

  Value* output =
      new_node->addOutput()->setType(matched_schema->return_types.at(0));
  return std::make_shared<SimpleValue>(output);
}

std::string PythonValue::kind() const {
  std::stringstream ss;
  ss << "python value of type '" << typeString(self) << "'";
  return ss.str();
}

std::vector<std::shared_ptr<SugaredValue>> PythonValue::asTuple(
    const SourceRange& loc,
    Function& m,
    const c10::optional<size_t>& size_hint) {
  const std::string type_str = typeString(self);
  std::stringstream ss;
  ss << kind() << " cannot be used as a tuple";
  checkForAddToConstantsError(ss);
  throw ErrorReport(loc) << ss.str();
}

std::shared_ptr<SugaredValue> PythonValue::attr(
    const SourceRange& loc,
    Function& m,
    const std::string& field) {
  const std::string type_str = typeString(self);
  std::stringstream ss;
  ss << "attribute lookup is not defined on " << kind();
  checkForAddToConstantsError(ss);
  throw ErrorReport(loc) << ss.str();
}

py::object PythonValue::getattr(
    const SourceRange& loc,
    const std::string& name) {
  try {
    return py::getattr(self, name.c_str());
  } catch (py::error_already_set& e) {
    throw ErrorReport(loc) << "object has no attribute " << name;
  }
}

void PythonValue::checkForAddToConstantsError(std::stringstream& ss) {
  auto nn = py::module::import("torch.nn");
  if (py::isinstance(self, nn.attr("ModuleList")) ||
      py::isinstance(self, nn.attr("Sequential"))) {
    ss << ". Did you forget to add it to __constants__? ";
  }
}

std::shared_ptr<SugaredValue> PythonModuleValue::attr(
    const SourceRange& loc,
    Function& m,
    const std::string& field) {
  py::object member = getattr(loc, field);
  // note: is_constant = true because we consider that global properties
  // on modules like math.pi or torch.float to be constants
  // eventhough it is possible, though rare, for someone to mutate them
  return toSugaredValue(member, m, loc, /*is_constant=*/true);
}

std::shared_ptr<SugaredValue> OverloadedMethodValue::call(
    const SourceRange& loc,
    Function& caller,
    at::ArrayRef<NamedValue> inputs,
    at::ArrayRef<NamedValue> attributes,
    size_t n_binders) {
  std::vector<NamedValue> new_inputs = inputs.vec();
  new_inputs.insert(new_inputs.begin(), module_);

  std::stringstream failure_messages;
  for (bool allow_conversions : {false, true}) {
    // clear previous error messages
    failure_messages.str("");
    for (const std::string& method_name : method_names_) {
      auto cls = module_->type()->expect<ClassType>();
      const auto fn = cls->getMethod(method_name);
      TORCH_INTERNAL_ASSERT(fn, "Expected class to have method ", method_name);
      auto match = tryMatchSchema(
          fn->getSchema(),
          loc,
          *caller.graph().get(),
          c10::nullopt,
          new_inputs,
          attributes,
          &failure_messages,
          allow_conversions);
      if (match) {
        return MethodValue(module_, method_name)
            .call(loc, caller, inputs, attributes, n_binders);
      }
    }
  }
  throw ErrorReport(loc) << failure_messages.str();
}

std::shared_ptr<SugaredValue> OverloadedFunctionValue::call(
    const SourceRange& loc,
    Function& caller,
    at::ArrayRef<NamedValue> inputs_,
    at::ArrayRef<NamedValue> attributes,
    size_t n_binders) {
  std::stringstream failure_messages;
  for (bool allow_conversions : {false, true}) {
    // clear previous error messages
    failure_messages.str("");
    for (const auto& compiled_overload : compiled_overloads_) {
      const auto matched_schema = tryMatchSchema(
          compiled_overload.function_->getSchema(),
          loc,
          *caller.graph(),
          c10::nullopt,
          inputs_,
          attributes,
          &failure_messages,
          allow_conversions);
      if (matched_schema) {
        return FunctionValue(compiled_overload)
            .call(loc, caller, inputs_, attributes, n_binders);
      }
    }
  }
  throw ErrorReport(loc) << failure_messages.str();
}

Value* ModuleValue::asValue(const SourceRange& loc, Function& m) {
  return self_;
}

static bool isModuleType(const TypePtr& type) {
  TORCH_INTERNAL_ASSERT(type);
  if (auto classType = type->cast<ClassType>()) {
    return classType->is_module();
  }
  return false;
}

IterableValuePtr ModuleValue::desugarModuleContainer(
    bool get_keys,
    bool get_values,
    const SourceRange& loc,
    Function& m) {
  std::vector<std::string> submoduleNames;
  const auto& selfType = concreteType_->getJitType();
  for (size_t i = 0; i < selfType->numAttributes(); ++i) {
    const auto& attrType = selfType->getAttribute(i);
    if (!attrType) {
      continue;
    }

    if (isModuleType(attrType)) {
      submoduleNames.push_back(selfType->getAttributeName(i));
    }
  }

  std::vector<SugaredValuePtr> keys;
  std::vector<SugaredValuePtr> values;
  for (const auto& name : submoduleNames) {
    auto name_v =
        std::make_shared<SimpleValue>(insertConstant(*m.graph(), name));
    Value* module_v = m.graph()->insertGetAttr(self_, name);
    auto mod_v = std::make_shared<ModuleValue>(
        module_v, concreteType_->findSubmoduleConcreteType(name));

    if (get_keys) {
      keys.push_back(name_v);
    }
    if (get_values) {
      values.push_back(mod_v);
    }
  }

  bool contains_module_list = true;
  int64_t len = submoduleNames.size();
  if (get_keys && !get_values) {
    return std::make_shared<SugaredTupleValue>(keys, true)->asIterable(loc, m);
  } else if (get_values && !get_keys) {
    return std::make_shared<SugaredTupleValue>(values, true)->asIterable(loc, m);
  } else if (get_values && get_keys) {
    auto key_list = std::make_shared<IterableValue>(std::make_shared<SugaredTupleValue>(keys, true), len, contains_module_list);
    auto value_list = std::make_shared<IterableValue>(std::make_shared<SugaredTupleValue>(values, true), len, contains_module_list);
    auto iterator = std::make_shared<IterableTree>();
    iterator->addChild(loc, key_list);
    iterator->addChild(loc, value_list);
    return iterator->asIterable(loc, m);
  } else {
    TORCH_INTERNAL_ASSERT(false);
  }
}

// This method controls how we desugar attribute lookups on ScriptModules.
std::shared_ptr<SugaredValue> ModuleValue::attr(
    const SourceRange& loc,
    Function& m,
    const std::string& field) {
  // 1. Look inside script::Module object for the field.
  const auto& selfType = concreteType_->getJitType();
  if (selfType->hasAttribute(field) && isModuleType(selfType->getAttribute(field))) {
    // ...if it's a submodule, return it as a new ModuleValue.
    const auto submoduleConcreteType =
        concreteType_->findSubmoduleConcreteType(field);
    TORCH_INTERNAL_ASSERT(submoduleConcreteType);
    return std::make_shared<ModuleValue>(
        m.graph()->insertGetAttr(self_, field), submoduleConcreteType);
  } else if (selfType->hasAttribute(field) || selfType->getMethod(field)) {
      // ...otherwise, methods, parameters, attributes, and buffers are all
      // first class so they get returned as SimpleValues
      return SimpleValue(self_).attr(loc, m, field);
  }

  // 2. Check if it's a user-provided constant property.
  if (auto constant = concreteType_->findConstant(field)) {
    // If it is, just insert the constant and return a SimpleValue for it.
    return toSugaredValue(*constant, m, loc, true);
  }

  // 3. Special case: for module dicts we manually desugar items(), keys(),
  // values() calls into the appropriate method.
  // TODO: These could be represented as first class methods probably.
  if (concreteType_->getIterableModuleKind() == IterableModuleKind::DICT) {
    if (field == "items" || field == "keys" || field == "values") {
      bool get_keys = false;
      bool get_values = false;
      if (field == "items") {
        get_keys = true;
        get_values = true;
      } else if (field == "values") {
        get_values = true;
      } else {
        get_keys = true;
      }
      return std::make_shared<ModuleDictMethod>(
          desugarModuleContainer(get_keys, get_values, loc, m), field);
    }
  }

  // 4. Check if this is the name of an overloaded method.

  // This can also be a call to a non-script module, or a plain
  // python method. If so return this as a python value.
  if (const auto overloads = concreteType_->findOverloads(field)) {
    return std::make_shared<OverloadedMethodValue>(self_, *overloads);
  }

  // 5. Check if it's a function attribute.
  if (const auto fnAttr = concreteType_->findFunctionAttribute(field)) {
    return std::make_shared<FunctionValue>(*fnAttr);
  }

  // 6. Check if it's a property of the original Python class that this
  // ScriptModule was derived from. The only class properties we handle are
  // methods.
  py::object unboundMethod = py::getattr(
      concreteType_->getPyClass(),
      field.c_str(),
      pybind11::cast<pybind11::none>(Py_None));
  if (py::isinstance<py::function>(unboundMethod)) {
    // For Python methods that we're trying to call directly, we need to bind
    // the method to a self. TODO say more about tis
    //
    // If the function is @ignored
    bool isIgnoredFn =
        py::cast<bool>(py::module::import("torch._jit_internal")
                           .attr("is_ignored_fn")(unboundMethod));
    if (isIgnoredFn) {
      // Create a generated ScriptModule type with module_ set as cpp_module
      auto boundMethod = py::module::import("torch.jit._recursive")
                             .attr("lazy_bind")(concreteType_, unboundMethod);
      TORCH_CHECK(py::isinstance<py::function>(boundMethod));
      auto rcb =
          py::module::import("torch._jit_internal")
              .attr("createResolutionCallbackFromClosure")(unboundMethod);
      return std::make_shared<PythonValue>(boundMethod, rcb, self_);
    }

    // If we reach here, it's because this is a "normal" method that just hasn't
    // been compiled yet (directly exported methods would have been returned by
    // step 1). Just compile it.
    auto stub =
        py::module::import("torch.jit._recursive")
            .attr("compile_unbound_method")(concreteType_, unboundMethod);
    TORCH_INTERNAL_ASSERT(!stub.is_none());
    return SimpleValue(self_).attr(loc, m, field);
  }

  // We've exhausted all possibilities. Bailout with a hint to the user.
  std::string hint;
  if (auto failureReason = concreteType_->findFailedAttribute(field)) {
    hint = *failureReason;
  }

  throw ErrorReport(loc) << "Module '" << selfType->name()->name() << "'"
                         << " has no attribute '" << field << "' " << hint;
}

IterableValuePtr ModuleValue::asIterable(
    const SourceRange& loc,
    Function& m) {
  const auto iterableModuleKind = concreteType_->getIterableModuleKind();
  if (iterableModuleKind == IterableModuleKind::NONE) {
    throw ErrorReport(loc) << "Only constant Sequential, ModueList, or ModuleDict can be used as an iterable";
  }

  // iterating over a dictionary returns the keys, iterating over a
  // list returns the values
  const bool get_keys = iterableModuleKind == IterableModuleKind::DICT;
  const bool get_values = iterableModuleKind == IterableModuleKind::LIST;
  return desugarModuleContainer(get_keys, get_values, loc, m);
}

void ModuleValue::setAttr(
    const SourceRange& loc,
    Function& m,
    const std::string& field,
    Value* newValue) {
  // Forward to SimpleValue::setAttr
  SimpleValue simple(self_);
  simple.setAttr(loc, m, field, newValue);
}

std::shared_ptr<SugaredValue> BooleanDispatchValue::call(
    const SourceRange& loc,
    Function& caller,
    at::ArrayRef<NamedValue> inputs,
    at::ArrayRef<NamedValue> attributes,
    size_t n_binders) {
  c10::optional<bool> result;
  Graph& graph = *(caller.graph());

  auto index = py::cast<size_t>(dispatched_fn_["index"]);
  auto arg_name = py::str(dispatched_fn_["arg_name"]);

  ErrorReport error(loc);
  if (index < inputs.size()) {
    // Dispatch flag is in arg list
    result = constant_as<bool>(inputs.at(index).value(graph));
    error << "Argument for boolean dispatch at position " << index
          << " was not constant";
  } else if (auto i = findInputWithName(arg_name, attributes)) {
    // Dispatch flag is in kwargs
    result = constant_as<bool>(attributes[*i].value(graph));
    error << "Keyword argument '" << arg_name
          << "' for boolean dispatch at position was not constant";
  } else {
    // Didn't find dispatch flag, so use default value
    result = py::cast<bool>(dispatched_fn_["default"]);
    TORCH_INTERNAL_ASSERT(result);
  }

  if (!result) {
    throw error;
  }

  std::shared_ptr<SugaredValue> value;
  if (*result) {
    value = toSugaredValue(dispatched_fn_["if_true"], caller, loc);
  } else {
    value = toSugaredValue(dispatched_fn_["if_false"], caller, loc);
  }
  return value->call(loc, caller, inputs, attributes, n_binders);
}
std::shared_ptr<SugaredValue> toSugaredValue(
    py::object obj,
    Function& m,
    SourceRange loc,
    bool is_constant) {

  // directly create SimpleValues when possible, because they are first-class
  // and can be re-assigned. Otherwise, this would be invalid:
  // f = python_constant
  // while ...
  //   f = f + 1
  auto& g = *m.graph();
  if (is_constant) {
    if (py::isinstance<py::bool_>(obj)) {
      return toSimple(g.insertConstant(py::cast<bool>(obj), loc));
    } else if (py::isinstance<py::int_>(obj)) {
      return toSimple(g.insertConstant(py::cast<int64_t>(obj), loc));
    } else if (py::isinstance<py::float_>(obj)) {
      return toSimple(g.insertConstant(py::cast<double>(obj), loc));
    } else if (py::isinstance<py::str>(obj)) {
      return toSimple(g.insertConstant(py::cast<std::string>(obj), loc));
    } else if (obj.is(py::none())) {
      return toSimple(g.insertConstant(IValue(), loc));
    } else if (THPDevice_Check(obj.ptr())) {
      auto device = reinterpret_cast<THPDevice*>(obj.ptr());
      return toSimple(g.insertConstant(device->device));
    } else if (THPLayout_Check(obj.ptr())) {
      auto layout = reinterpret_cast<THPLayout*>(obj.ptr());
      const auto v = static_cast<int64_t>(layout->layout);
      return toSimple(g.insertConstant(v, loc));
    } else if (THPDtype_Check(obj.ptr())) {
      auto dtype = reinterpret_cast<THPDtype*>(obj.ptr());
      const auto v = static_cast<int64_t>(dtype->scalar_type);
      return toSimple(g.insertConstant(v, loc));
    } else if (THPQScheme_Check(obj.ptr())) {
      auto qscheme = reinterpret_cast<THPQScheme*>(obj.ptr());
      const auto v = static_cast<uint8_t>(qscheme->qscheme);
      return toSimple(g.insertConstant(v, loc));
    } else if (THPLayout_Check(obj.ptr())) {
      auto layout = reinterpret_cast<THPLayout*>(obj.ptr());
      const auto l = static_cast<int8_t>(layout->layout);
      return toSimple(g.insertConstant(l, loc));
    } else if (py::isinstance<py::tuple>(obj)) {
      py::tuple tup = obj;
      std::vector<std::shared_ptr<SugaredValue>> result;
      result.reserve(tup.size());
      for (py::handle t : tup) {
        py::object obj = py::reinterpret_borrow<py::object>(t);
        result.push_back(toSugaredValue(obj, m, loc, true));
      }
      bool contains_module_list = false; // Python Tuples can't contain module list
      return std::make_shared<SugaredTupleValue>(result, contains_module_list);
    }
  }

  if (auto callee = as_function(obj)) {
    return std::make_shared<FunctionValue>(callee->function_);
  } else if (py::isinstance<py::module>(obj)) {
    return std::make_shared<PythonModuleValue>(obj);
  } else if (obj.ptr() == py::module::import("torch.jit").attr("_fork").ptr()) {
    return SpecialFormValue::create(prim::fork);
  } else if (
      obj.ptr() == py::module::import("torch.jit").attr("annotate").ptr()) {
    return SpecialFormValue::create(prim::annotate);
  } else if (auto callee = as_module(obj)) {
    throw ErrorReport(loc) << "Cannot call a ScriptModule that is not"
                           << " a submodule of the caller";
  }

  py::object builtin_name =
      py::module::import("torch.jit").attr("_find_builtin")(obj);
  if (!builtin_name.is_none()) {
    return std::make_shared<BuiltinFunction>(
        Symbol::fromQualString(py::str(builtin_name)), c10::nullopt);
  }

  if (py::isinstance<py::function>(obj)) {
    if (typeString(obj) == "builtin_function_or_method") {
      throw ErrorReport(loc) << "Python builtin " << py::str(obj)
                             << " is currently not supported in Torchscript";
    }
  }

  py::object dispatched_fn =
      py::module::import("torch.jit").attr("_try_get_dispatched_fn")(obj);
  if (!dispatched_fn.is_none()) {
    return std::make_shared<BooleanDispatchValue>(std::move(dispatched_fn));
  }

  py::bool_ isClass = py::module::import("inspect").attr("isclass")(obj);
  if (py::cast<bool>(isClass)) {
    py::str qualifiedName =
        py::module::import("torch.jit").attr("_qualified_name")(obj);
    auto pyCu = get_python_cu();
    auto qualname = c10::QualifiedName(qualifiedName);
    if (auto classType = pyCu->get_class(qualname)) {
      return std::make_shared<ClassValue>(classType);
    } else {
      // If we can't get the source code for the type, it's implemented in C and
      // probably part of the standard library, so give up and leave it as a
      // call to Python
      bool can_compile_class =
          py::cast<bool>(py::module::import("torch._jit_internal")
                             .attr("can_compile_class")(obj));
      if (can_compile_class) {
        // Register class
        auto rcb = py::module::import("torch._jit_internal")
                       .attr("createResolutionCallbackForClassMethods")(obj);

        {
          // We're starting a new compilation, so update the error call stack in
          // case it fails
          ErrorReport::CallStack stack(qualname.name());
          ErrorReport::CallStack::update_pending_range(loc);

          py::module::import("torch.jit")
              .attr("_compile_and_register_class")(obj, rcb, qualifiedName);
        }

        // Return class
        auto newClassType = pyCu->get_class(qualname);
        AT_ASSERT(
            newClassType,
            "Class '",
            qualifiedName,
            "' should have been compiled but was not");
        return std::make_shared<ClassValue>(newClassType);
      }
    }
  }

  py::bool_ isFunction = py::module::import("inspect").attr("isfunction")(obj);
  if (py::cast<bool>(isFunction)) {
    auto overloads =
        py::module::import("torch.jit").attr("_get_overloads")(obj);
    if (!overloads.is_none()) {
      auto compiled_fns = py::cast<std::vector<StrongFunctionPtr>>(overloads);
      return std::make_shared<OverloadedFunctionValue>(std::move(compiled_fns));
    }

    auto compiled_fn =
        py::module::import("torch.jit._recursive").attr("try_compile_fn")(obj, loc);
    if (auto callee = as_function(compiled_fn)) {
      return std::make_shared<FunctionValue>(*callee);
    }
  }

  py::bool_ isMethod = py::module::import("inspect").attr("ismethod")(obj);
  // methods here have been explicitly annotated to not be compiled,
  // so they do not have the same overload and compile checks as for functions
  if (isFunction || isMethod) {
    auto rcb = py::module::import("torch._jit_internal")
                   .attr("createResolutionCallbackFromClosure")(obj);
    return std::make_shared<PythonValue>(obj, rcb);
  }

  return std::make_shared<PythonValue>(obj);
}
} // namespace script
} // namespace jit
} // namespace torch
