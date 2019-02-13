//==-- llvm/Support/FileCheck.h ---------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// API modified from llvm::FileCheck

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/source_range.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {
namespace testing {

enum CheckType {
  CHECK,
  CHECK_NEXT,
  CHECK_SAME,
  CHECK_NOT,
  CHECK_COUNT,
  CHECK_DAG,
};

struct Check {
  Check(
      CheckType type,
      std::string str,
      c10::optional<size_t> count = c10::nullopt)
      : type_(type), search_str_(std::move(str)) {
    count_ = count;
  };
  //
  // void setSourceLocation(SourceRange sl) {
  //   source_range_ = std::move(sl);
  // }

  CheckType type_;
  c10::optional<size_t> count_;
  const std::string search_str_;
};

namespace {
static std::string escapeString(const std::string& input) {
  std::string s = input;
  std::vector<char> search = {'\n', '\t', '\v'};
  std::vector<std::string> replace = {"\\n", "\\t", "\\v"};
  for (size_t i = 0; i < search.size(); i++) {
    for (size_t i = 0; i < search.size(); i++) {
      size_t pos = s.find(search[i]);
      while (pos != std::string::npos) {
        s.replace(pos, 1, replace[i]);
        pos = s.find(search[i], pos + 1);
      }
    }
  }
  return s;
}

size_t assertFind(
    SourceRange range,
    const std::string& sub,
    size_t start,
    std::function<void(std::ostream& out)> extra_msg = nullptr) {
  auto pos = range.file().find(sub, start);
  if (pos == std::string::npos || (pos + sub.size()) > range.end()) {
    auto range =
        SourceRange(std::make_shared<std::string>(file), start, sub.size());
    std::stringstream ss;
    ss << "Expected to find '" << escapeString(sub)
       << "' but did not find it\n";
    range.highlight(ss);
    if (extra_msg)
      extra_msg(ss);
    throw std::runtime_error(ss.str());
  }
  return pos;
}

size_t assertFind(
    std::shared_ptr<std::string> file,
    const std::string& sub,
    size_t start,
    const Check& check) {
  return assertFind(file, sub, start, [&](std::ostream& out) {
    // out << "From the check defined\n";
    // check.source_range_->highlight(out);
  });
}

void assertNotFind(
    SourceRange range,
    const std::string& sub,
    const Check& check) {
  auto pos = range.file().find(sub);
  if (pos != std::string::npos && (pos + sub.size()) < range.end()) {
    // auto range =
    //     SourceRange(std::make_shared<std::string>(file), pos, sub.size() + pos);
    std::stringstream ss;
    ss << "Expected to not find '" << escapeString(sub) << "' but found it\n";
    range.highlight(ss);
    // ss << "From the check defined\n";
    // check.source_range_->highlight(ss);
    throw std::runtime_error(ss.str());
  }
}
} // namespace

struct FileCheckImpl {
  TORCH_API explicit FileCheckImpl() {}

  TORCH_API void checkFile(const std::string& test_file) {
    has_run = true;
    doChecks(test_file);
  }

  TORCH_API void addCheck(CheckType type, const std::string& s, c10::optional<size_t> count = c10::nullopt) {
   Check check(type, s, count);

   // consecutive CHECK_DAGs & CHECK_NOTs need to be evaluated as a group
   if (groups.size() == 0 || (type != CHECK_NOT && type != CHECK_DAG)) {
     groups.push_back({check});
   } else {
     auto& last_group = groups.back();
     if (last_group.at(0).type_ == type) {
       last_group.push_back(check);
     } else {
       groups.push_back({check});
     }
   }

   has_run = false;
  }

  bool has_run;

 private:

  // consecutive CHECK_DAGs & CHECK_NOTs need to be evaluated as a group
  void makeGroups(std::vector<Check> input) {
    for (size_t i = 0; i < input.size(); ++i) {
      std::vector<Check> group = {input[i]};
      CheckType type = input[i].type_;
      if (type != CHECK_NOT && type != CHECK_DAG) {
        groups.push_back(group);
        continue;
      }
      while (i + 1 < input.size() && input[i + 1].type_ == type) {
        ++i;
        group.push_back(input[i]);
      }
      groups.push_back(group);
    }
  }

  void doCheckNot(
      const std::vector<Check>& nots,
      std::shared_ptr<std::string> file,
      SourceRange prev,
      SourceRange next) {
    auto start = prev.end(); // inclusive
    auto end = next.start(); // exclusive
    if (end < start) {
      return;
    }
    const auto& substr = file.substr(start, end - start);
    for (const auto& check : nots) {
      AT_ASSERT(check.type_ == CHECK_NOT);
      assertNotFind(substr, check.search_str_, check);
    }
  }

  SourceRange matchDagGroup(
      const std::vector<Check>& group,
      std::shared_ptr<std::string> test_file,
      SourceRange prev) {
    size_t group_beg = std::string::npos;
    size_t group_end = 0;

    AT_ASSERT(groups.size() != 0);
    for (const auto& check : group) {
      AT_ASSERT(check.type_ == group[0].type_);
      auto pos = assertFind(test_file, check.search_str_, prev.end(), check);
      group_beg = std::min(pos, group_beg);
      group_end = std::max(pos + check.search_str_.size(), group_end);
    }

    return SourceRange(test_file, group_beg, group_end);
  }

  SourceRange matchGroup(
      const std::vector<Check>& group,
      std::shared_ptr<std::string> test_file,
      SourceRange prev) {
    AT_ASSERT(group.size() != 0);
    CheckType type = group[0].type_;

    if (type == CHECK_DAG) {
      return matchDagGroup(group, test_file, prev);
    }
    AT_ASSERT(type != CHECK_NOT);
    AT_ASSERT(group.size() == 1);

    const auto& check = group[0];
    size_t start_range = prev.end();
    size_t end_range = start_range;

    switch (check.type_) {
      case CHECK: {
        start_range =
            assertFind(test_file, check.search_str_, start_range, check);
        end_range = start_range + check.search_str_.size();
      } break;
      case CHECK_SAME: {
        auto pos = assertFind(test_file, check.search_str_, start_range, check);
        assertNotFind(SourceRange(test_file, prev.end(), pos), "\n", check);
        start_range = pos;
        end_range = pos + check.search_str_.size();
      } break;
      case CHECK_NEXT: {
        auto line_end = assertFind(test_file, "\n", start_range, check);
        auto pos =
            assertFind(test_file, check.search_str_, line_end + 1, check);
        assertNotFind(
            test_file.substr(line_end + 1, pos - (line_end + 1)), "\n", check);
        start_range = pos;
        end_range = pos + check.search_str_.size();
      } break;
      case CHECK_COUNT: {
        auto group_start_range = std::string::npos;
        AT_ASSERT(check.count_ && *check.count_ != 0);
        for (size_t i = 0; i < *check.count_; ++i) {
          start_range =
              assertFind(test_file, check.search_str_, start_range, check);
          group_start_range = std::min(start_range, group_start_range);
          end_range = start_range + check.search_str_.size();
          start_range = end_range;
        }
        start_range = group_start_range;
      } break;
      case CHECK_DAG: {
        AT_ASSERT(false);
      } break;
      case CHECK_NOT: {
        AT_ASSERT(false);
      } break;
    }
    return SourceRange(test_file, start_range, end_range);
  }

  void doChecks(std::shared_ptr<std::string> test_file) {
    SourceRange prev(test_file, 0, 0);
    for (size_t i = 0; i < groups.size(); i++) {
      const auto& curr_group = groups[i];
      CheckType type = curr_group.at(0).type_;
      if (type != CHECK_NOT) {
        prev = matchGroup(curr_group, test_file, prev);
      } else {
        if (i + 1 < groups.size()) {
          const auto& next_group = groups[i + 1];
          AT_ASSERT(next_group.at(0).type_ != CHECK_NOT);
          SourceRange after_not = matchGroup(next_group, test_file, prev);
          doCheckNot(curr_group, test_file, prev, after_not);
          prev = after_not;
          ++i; // already checked the group after
        } else {
          SourceRange end_of_file(test_file, test_file->size() + 1, test_file->size() + 1);
          doCheckNot(curr_group, test_file, prev, end_of_file);
        }
      }
    }
  }

  std::vector<Check> checks;
  std::shared_ptr<std::string> check_file;
  std::vector<std::vector<Check>> groups;
};

FileCheck::FileCheck() : fcImpl(new FileCheckImpl()) {};

FileCheck::~FileCheck() {
  if (!fcImpl->has_run) {
    std::cout << "You have not run this instance of FileCheck!";
  }
  fcImpl.reset();
};

void FileCheck::checkFile(const std::string& test_file) {
  fcImpl->checkFile(test_file);
};

FileCheck* FileCheck::check(const std::string& str) {
  fcImpl->addCheck(CHECK, str);
  return this;
}

FileCheck* FileCheck::check_not(const std::string& str) {
  fcImpl->addCheck(CHECK, str);
  return this;
}

FileCheck* FileCheck::check_same(const std::string& str) {
  fcImpl->addCheck(CHECK_SAME, str);
  return this;
}

FileCheck* FileCheck::check_next(const std::string& str) {
  fcImpl->addCheck(CHECK_SAME, str);
  return this;
}

FileCheck* FileCheck::check_count(const std::string& str, size_t count) {
  fcImpl->addCheck(CHECK_COUNT, str, count);
  return this;
}

FileCheck* FileCheck::check_dag(const std::string& str) {
  fcImpl->addCheck(CHECK_DAG, str);
  return this;
}





} // namespace testing
} // namespace jit
} // namespace torch
