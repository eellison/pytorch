#!/usr/bin/env python3
"""
HOSTTRACE_SYNC_API: CUDA host code must not call a synchronous memory API, and
must not reach the host-tracing recorder's private interface.

A traceable host runs under a stream capture that sees every kernel, async
copy and memset it issues. A synchronous cudaMemcpy / cudaMemcpyToSymbol /
cudaMemset (or the driver cuMemcpy* / cuMemset* forms) is invisible to a
capture in every mode (CUDA 13.0): it runs at the trace and never in the
replayed graph. Such a call is a host-contract violation
(aten/src/ATen/cuda/host_trace/Recorder.h), and a host that makes one is
already wrong under a plain CUDA graph capture. Use the asynchronous form on
the current stream, or move a one-time upload to the warm-up call trace()
makes before the symbolic run. Checked: every translation unit under the CUDA
host directories (a host reached through its own headers included), plus any
file that includes the host_trace headers. One-time initializations are listed
in host_trace_sync_api_allowlist.txt beside this file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys


LINTER_CODE = "HOSTTRACE_SYNC_API"
_HOST_DIRS = (
    "aten/src/ATen/native/cuda/",
    "aten/src/ATen/native/transformers/cuda/",
    "aten/src/ATen/native/sparse/cuda/",
    "aten/src/ATen/native/nested/cuda/",
    "aten/src/ATen/cuda/",
    "torch/csrc/cuda/",
)
_RECORDER_SOURCES = ("aten/src/ATen/cuda/host_trace/", "torch/csrc/cuda/HostTrace.cpp")
_ALLOWLIST = os.path.join(
    os.path.dirname(__file__), "host_trace_sync_api_allowlist.txt"
)
_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]ATen/cuda/host_trace/', re.MULTILINE)
_NAMES = (
    r"cudaMemcpy(?:2D(?:ToArray|FromArray|ArrayToArray)?|3D(?:Peer)?|ToArray|FromArray|"
    r"ArrayToArray|ToSymbol|FromSymbol|Peer)?|cudaMemset(?:2D|3D)?|"
    r"cuMemcpy(?:HtoD|DtoH|DtoD|HtoA|AtoH|AtoA|AtoD|DtoA|2D(?:Unaligned)?|3D(?:Peer)?|Peer)?|"
    r"cuMemset(?:D2D|D)(?:8|16|32)"
)
# a call, or the address of the function; the Async forms never match (\b)
_SYNC = re.compile(r"(&\s*)?\b(" + _NAMES + r")(?:_v2)?\b(\s*\()?")
_PRIVATE = re.compile(r"\b(HintsInternal|host_trace::hooks)\b")
# comments and string literals blanked out, newlines kept so line numbers hold
_BLANK = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"', re.DOTALL)


def _norm(path: str) -> str:
    return "/" + path.replace(os.sep, "/").lstrip("./")


def _allowed(path: str) -> set[str]:
    out = set()
    if os.path.isfile(_ALLOWLIST):
        with open(_ALLOWLIST, encoding="utf-8") as f:
            for line in f:
                entry = line.split("#", 1)[0].split()
                if len(entry) == 2 and _norm(path).endswith("/" + entry[0]):
                    out.add(entry[1])
    return out


def _message(path: str, code: str, pos: int, name: str, desc: str) -> dict:
    return {
        "path": path,
        "line": code.count("\n", 0, pos) + 1,
        "char": pos - code.rfind("\n", 0, pos),
        "code": LINTER_CODE,
        "severity": "error",
        "name": name,
        "original": None,
        "replacement": None,
        "description": desc,
    }


def check_file(path: str) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    code = _BLANK.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    norm = _norm(path)
    host = any("/" + d in norm for d in _HOST_DIRS) or _INCLUDE.search(code)
    recorder = any("/" + d in norm for d in _RECORDER_SOURCES)
    out: list[dict] = []
    if host:
        allowed = _allowed(path)
        for m in _SYNC.finditer(code):
            if not (m.group(1) or m.group(3)) or m.group(2) in allowed:
                continue
            out.append(
                _message(
                    path,
                    code,
                    m.start(2),
                    "sync-memory-api-in-cuda-host",
                    f"`{m.group(2)}` in CUDA host code: a synchronous memory API call "
                    "is invisible to a stream capture and never replays. Use the "
                    "asynchronous form on the current stream, or move a one-time "
                    "upload to the warm-up before the trace (see "
                    "ATen/cuda/host_trace/Recorder.h).",
                )
            )
    if not recorder:
        for m in _PRIVATE.finditer(code):
            out.append(
                _message(
                    path,
                    code,
                    m.start(),
                    "recorder-private-interface",
                    f"`{m.group(1)}` is the host-tracing recorder's private "
                    "interface; a host reads a value with guard_int or guards on it.",
                )
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("filenames", nargs="+")
    for filename in parser.parse_args().filenames:
        for message in check_file(filename):
            print(json.dumps(message), flush=True)


if __name__ == "__main__":
    sys.exit(main())
