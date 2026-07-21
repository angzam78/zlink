#!/usr/bin/env python3
"""
zlink CUDA Driver API Code Generator
=====================================

Generates comprehensive CUDA Driver API shim/server code from an annotated
function database (JSON). Inspired by Lupine's codegen approach but simpler:
instead of parsing C++ headers with annotations, we use a structured JSON
database that describes every CUDA function's signature and RPC semantics.

OUTPUTS:
  gen_api.hpp       — RPC struct declarations, function declarations, zpp::bits bindings
  gen_shim.inc      — Client-side shim function bodies (extern "C" cu* exports)
  gen_server.inc    — Server-side handler bodies (calls real CUDA on GPU)

DESIGN:
  Each CUDA function is annotated with:
    - category: "barrier" | "enqueued" | "readback"
        barrier  = must wait for return value before next call
        enqueued = can be pipelined (most functions)
        readback = flush pipeline + get data back
    - handle_produces: list of output params that create opaque handles
        (CUcontext, CUstream, CUevent, CUmodule, CUfunction, etc.)
    - handle_consumes: list of input params that are opaque handles
    - devptr_params: list of params that are CUdeviceptr (use ptr_map)
    - host_sync_params: list of params whose data must sync to server first
    - host_readback: list of params that need readback from server after
    - local_only: handled entirely on client side, no RPC needed
    - disabled: skip code generation entirely
    - stub_return: return this CUDA error code instead of making RPC

  The generated code integrates with existing hand-written infrastructure:
    - handle_map (register_handle, translate_handle, unregister_handle)
    - ptr_map (map, to_remote, unmap)
    - host_sync/readback (sync_host_to_server, readback_from_server)
    - RPC dispatch (cuda_rpc_call<index>(args...))
"""

import json
import sys
import os
import re
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════
# CUDA type system — maps CUDA types to C++ serializable types
# ══════════════════════════════════════════════════════════════════════════

# Maps CUDA C types to our shim's C++ types (for the RPC API declarations)
CUDA_TYPE_MAP = {
    # Return type
    "CUresult": "int32_t",

    # Opaque handles — represented as uint64_t for RPC
    "CUcontext":   "uint64_t",
    "CUstream":    "uint64_t",
    "CUevent":     "uint64_t",
    "CUmodule":    "uint64_t",
    "CUfunction":  "uint64_t",
    "CUgraph":     "uint64_t",
    "CUgraphNode": "uint64_t",
    "CUgraphExec": "uint64_t",
    "CUmipmappedArray": "uint64_t",
    "CUarray":     "uint64_t",
    "CUsurfref":   "uint64_t",
    "CUtexref":    "uint64_t",
    "CUextMemory": "uint64_t",
    "CUextSemaphore": "uint64_t",
    "CUkernel":    "uint64_t",
    "CUlibrary":   "uint64_t",
    "CUkernelNode": "uint64_t",
    "CUmemGenericAllocationHandle": "uint64_t",
    "CUexternalMemory": "uint64_t",
    "CUexternalSemaphore": "uint64_t",
    "CUlaunchConfig": "uint64_t",
    "CUlaunchAttribute": "uint64_t",
    "CUaccessPolicyWindow": "uint64_t",
    "CUfunc_cache": "uint64_t",
    "CUsharedconfig": "uint64_t",

    # Device pointer — also uint64_t
    "CUdeviceptr": "uint64_t",

    # Integer types
    "CUdevice":    "int32_t",
    "int":         "int32_t",
    "unsigned int": "uint32_t",
    "size_t":      "uint64_t",
    "unsigned long long": "uint64_t",
    "long long":   "int64_t",
    "long":        "int64_t",
    "unsigned long": "uint64_t",
    "int32_t":     "int32_t",
    "uint32_t":    "uint32_t",
    "int64_t":     "int64_t",
    "uint64_t":    "uint64_t",
    "cuuint64_t":  "uint64_t",
    "cuuint32_t":  "uint32_t",

    # Float types
    "float":       "float",
    "double":      "double",

    # Enum types — all passed as int32_t
    "CUmem_alloc_granularity_flags": "uint64_t",
    "CUmem_access_flags":            "uint64_t",
    "CUmem_allocation_handle_type":  "uint64_t",
    "CUmem_allocation_type":         "uint64_t",
    "CUmem_location_type":           "uint64_t",
    "CUmem_advise":                  "uint64_t",
    "CUmem_range_attribute":         "uint64_t",
    "CUarray_format":                "uint32_t",
    "CUaddress_mode":                "uint32_t",
    "CUfilter_mode":                 "uint32_t",
    "CUresourcetype":                "uint32_t",
    "CUgraphicsRegisterFlags":       "uint32_t",
    "CUgraphicsMapResourceFlags":    "uint32_t",
    "CUgraphicsSetResourceFlags":    "uint32_t",
    "CUlimit":                       "uint32_t",
    "CUresult":                      "int32_t",
    "CUctx_flags":                   "uint32_t",
    "CUevent_flags":                 "uint32_t",
    "CUstream_flags":                "uint32_t",
    "CUoccupancy_flags":             "uint32_t",
    "CUmem_host_register_flags":     "uint32_t",
    "CUmem_host_alloc_flags":        "uint32_t",
    "CUmem_attach_flags":            "uint32_t",
    "CUarray_cubemap_face":          "uint32_t",
    "CUstream_callback":             "uint64_t",  # function pointer → uint64
    "CUoutput_mode":                 "uint32_t",
    "CUprofiler_trigger_mode":       "uint32_t",
    "CUgraphNodeType":               "uint32_t",
    "CUgraphExecUpdateResult":       "uint32_t",
    "CUgraphDebugDotFlags":          "uint64_t",
    "CUlaunchMemSyncDomain":         "uint32_t",
    "CUlaunchMemSyncDomainFlags":    "uint64_t",
    "CUaccessProperty":              "uint32_t",
    "CUaccessOrder":                 "uint32_t",

    # Pointer types — void*/char* etc become uint64_t for RPC
    "void*":        "uint64_t",
    "const void*":  "uint64_t",
    "char*":        "uint64_t",
    "const char*":  "uint64_t",   # passed as std::string for RPC
    "void**":       "uint64_t",
    "CUcontext*":   "uint64_t",
    "CUstream*":    "uint64_t",
    "CUevent*":     "uint64_t",
    "CUmodule*":    "uint64_t",
    "CUfunction*":  "uint64_t",
    "CUgraph*":     "uint64_t",
    "CUgraphNode*": "uint64_t",
    "CUgraphExec*": "uint64_t",
    "CUarray*":     "uint64_t",
    "CUmipmappedArray*": "uint64_t",
    "CUsurfref*":   "uint64_t",
    "CUtexref*":    "uint64_t",
    "CUdeviceptr*": "uint64_t",
    "CUdevice*":    "uint64_t",
    "CUexternalMemory*": "uint64_t",
    "CUexternalSemaphore*": "uint64_t",
    "CUmemGenericAllocationHandle*": "uint64_t",
    "CUgraphicsResource*": "uint64_t",
    "size_t*":      "uint64_t",
    "int*":         "uint64_t",
    "unsigned int*": "uint64_t",
    "float*":       "uint64_t",
    "unsigned long long*": "uint64_t",
    "long long*":   "uint64_t",
    "char**":       "uint64_t",
}

# Handle types — these are opaque pointers that need virtual handle translation
HANDLE_TYPES = {
    "CUcontext", "CUstream", "CUevent", "CUmodule", "CUfunction",
    "CUgraph", "CUgraphNode", "CUgraphExec",
    "CUmipmappedArray", "CUarray", "CUsurfref", "CUtexref",
    "CUexternalMemory", "CUexternalSemaphore",
    "CUkernel", "CUlibrary", "CUkernelNode",
    "CUgraphicsResource",
    "CUmemGenericAllocationHandle",
}

# Handle pointer types — output params that produce handles
HANDLE_PTR_TYPES = {f"{t}*" for t in HANDLE_TYPES}

# ══════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════

def snake_case(name: str) -> str:
    """Convert camelCase or PascalCase to snake_case."""
    # Handle cu prefix specially: cuMemAlloc → mem_alloc
    if name.startswith("cu"):
        name = name[2:]
    s1 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()

def pascal_case(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return ''.join(word.capitalize() for word in name.split('_'))

def cpp_rpc_type(cuda_type: str) -> str:
    """Map a CUDA C type to the C++ type used in RPC declarations."""
    t = cuda_type.strip()
    if t in CUDA_TYPE_MAP:
        return CUDA_TYPE_MAP[t]
    # Handle const char* specially → std::string for RPC
    if t == "const char*":
        return "std::string"
    # Handle pointer types
    if t.endswith("*"):
        return "uint64_t"
    # Fallback
    return "uint64_t"

def is_handle_type(cuda_type: str) -> bool:
    """Check if a CUDA type is an opaque handle type."""
    base = cuda_type.strip().rstrip("*").replace("const ", "")
    return base in HANDLE_TYPES

def is_output_handle(param: dict) -> bool:
    """Check if a param is an output handle (pointer to handle type)."""
    return param.get("direction") == "out" and is_handle_type(param["type"])

def is_input_handle(param: dict) -> bool:
    """Check if a param is an input handle (handle type, not pointer)."""
    t = param["type"].strip()
    base = t.replace("const ", "")
    return base in HANDLE_TYPES and "*" not in t

def is_devptr(param: dict) -> bool:
    """Check if a param is a device pointer (CUdeviceptr)."""
    return param["type"].strip() == "CUdeviceptr"

def is_host_ptr(param: dict) -> bool:
    """Check if a param points to host memory that needs syncing."""
    return param.get("host_sync", False)

def needs_readback(param: dict) -> bool:
    """Check if a param needs readback from server after the call."""
    return param.get("readback", False)

# ══════════════════════════════════════════════════════════════════════════
# Code generators
# ══════════════════════════════════════════════════════════════════════════

class CudaCodegen:
    def __init__(self, db_path: str):
        with open(db_path, 'r') as f:
            self.db = json.load(f)
        self.functions = self.db["functions"]
        # Filter out disabled functions
        self.active_functions = [f for f in self.functions if not f.get("disabled", False)]
        # Build index mapping
        self.rpc_index = {}
        idx = 0
        for f in self.functions:
            if not f.get("disabled", False):
                self.rpc_index[f["name"]] = idx
                idx += 1

    def total_active(self) -> int:
        return len(self.active_functions)

    # ── Generate gen_api.hpp ──────────────────────────────────────────
    def gen_api_hpp(self) -> str:
        lines = []
        lines.append("#pragma once")
        lines.append("// AUTO-GENERATED by codegen/codegen.py — DO NOT EDIT")
        lines.append(f"// Contains {self.total_active()} CUDA Driver API function RPC declarations")
        lines.append("//")
        lines.append("// This file is included by both the client shim and the server.")
        lines.append("// It defines the RPC message types (request/response structs) and")
        lines.append("// the zpp::bits::rpc<> binding that maps function indices to handlers.")
        lines.append("")
        lines.append("#include <zpp_bits.h>")
        lines.append("#include <cstdint>")
        lines.append("#include <string>")
        lines.append("")
        lines.append("namespace cuda_gen {")
        lines.append("")

        # Generate return structs and function declarations
        for func in self.active_functions:
            name = func["name"]
            rpc_name = snake_case(name)
            lines.append(f"// ── {name} | {func.get('category', 'enqueued')} ──")
            lines.append(self._gen_return_struct(func))
            lines.append(self._gen_func_declaration(func))
            lines.append("")

        # Generate RPC binding
        lines.append("// ══════════════════════════════════════════════════════════════")
        lines.append("// RPC binding — maps function indices to handlers")
        lines.append("// ══════════════════════════════════════════════════════════════")
        lines.append("using cuda_gen_rpc = zpp::bits::rpc<")
        for i, func in enumerate(self.active_functions):
            name = func["name"]
            rpc_name = snake_case(name)
            idx = self.rpc_index[name]
            comma = "," if i < len(self.active_functions) - 1 else ""
            lines.append(f"    zpp::bits::bind<&cuda_gen::{rpc_name}, {idx}>{comma}  // {name}")
        lines.append(">;")
        lines.append("")

        # Generate named indices
        lines.append("// ── Named function indices ───────────────────────────────")
        lines.append("namespace func_index {")
        for func in self.active_functions:
            name = func["name"]
            rpc_name = snake_case(name)
            idx = self.rpc_index[name]
            lines.append(f"    constexpr int {rpc_name} = {idx};  // {name}")
        lines.append("}")
        lines.append("")

        lines.append("} // namespace cuda_gen")
        return '\n'.join(lines)

    def _gen_return_struct(self, func: dict) -> str:
        """Generate the return struct for a function."""
        name = func["name"]
        rpc_name = snake_case(name)
        struct_name = f"{pascal_case(rpc_name)}Ret"
        lines = [f"struct {struct_name} {{"]

        # Always include result
        lines.append("    int32_t result;")

        # Collect field names to avoid duplicates
        seen_fields = set()
        handle_produces = set(func.get("handle_produces", []))
        devptr_produces = set(func.get("devptr_produces", []))

        # Add output fields — skip those covered by handle/devptr produces
        for param in func.get("params", []):
            if param.get("direction") == "out" or param.get("direction") == "inout":
                field_name = param["name"]
                if field_name in handle_produces or field_name in devptr_produces:
                    continue  # Will be added below as uint64_t
                if field_name in seen_fields:
                    continue
                seen_fields.add(field_name)

                ptype_stripped = param["type"].strip()
                if ptype_stripped == "const char*" and param.get("direction") == "out":
                    field_type = "std::string"
                elif ptype_stripped == "CUdevice*":
                    field_type = "int32_t"
                elif ptype_stripped == "size_t*":
                    field_type = "uint64_t"
                elif ptype_stripped == "unsigned int*":
                    field_type = "uint32_t"
                elif ptype_stripped == "int*":
                    field_type = "int32_t"
                elif ptype_stripped == "float*":
                    field_type = "float"
                elif ptype_stripped == "unsigned long long*":
                    field_type = "uint64_t"
                elif ptype_stripped.endswith("*"):
                    field_type = "uint64_t"
                else:
                    field_type = cpp_rpc_type(ptype_stripped)

                lines.append(f"    {field_type} {field_name};")

        # Add handle produces
        for hp in func.get("handle_produces", []):
            if hp not in seen_fields:
                lines.append(f"    uint64_t {hp};  // PRODUCES handle")
                seen_fields.add(hp)

        # Add devptr produces
        for dp in func.get("devptr_produces", []):
            if dp not in seen_fields:
                lines.append(f"    uint64_t {dp};  // PRODUCES devptr")
                seen_fields.add(dp)

        lines.append("};")
        return '\n'.join(lines)

    def _gen_func_declaration(self, func: dict) -> str:
        """Generate the RPC function declaration."""
        name = func["name"]
        rpc_name = snake_case(name)
        struct_name = f"{pascal_case(rpc_name)}Ret"

        # Build parameter list (input params only for RPC)
        params = []
        for param in func.get("params", []):
            if param.get("direction", "in") in ("in", "inout"):
                ptype = cpp_rpc_type(param["type"])
                pname = param["name"]
                # Special: const char* input → std::string
                if param["type"].strip() == "const char*" and param.get("direction") != "out":
                    ptype = "std::string"
                params.append(f"{ptype} {pname}")

        # Handle produces as extra output (not input params)
        # Devptr produces are output values

        param_str = ", ".join(params)
        return f"{struct_name} {rpc_name}({param_str});"

    # ── Generate gen_shim.inc ─────────────────────────────────────────
    def gen_shim_inc(self) -> str:
        """Generate the client-side shim function bodies."""
        lines = []
        lines.append("// AUTO-GENERATED by codegen/codegen.py — DO NOT EDIT")
        lines.append(f"// Contains {self.total_active()} CUDA Driver API shim function bodies")
        lines.append("// Included inside extern \"C\" { ... } in cuda_shim_codegen.cpp")
        lines.append("")

        for func in self.active_functions:
            if func.get("local_only", False):
                lines.append(self._gen_local_only_shim(func))
            elif func.get("stub_return") is not None:
                lines.append(self._gen_stub_shim(func))
            else:
                lines.append(self._gen_rpc_shim(func))
            lines.append("")

        return '\n'.join(lines)

    def _gen_local_only_shim(self, func: dict) -> str:
        """Generate a shim for a function handled entirely on client side."""
        name = func["name"]
        params = func.get("params", [])
        param_strs = []
        arg_names = []

        for p in params:
            param_strs.append(f"{p['type']} {p['name']}")
            arg_names.append(p['name'])

        lines = [f"// {name} — local_only (no RPC)"]
        lines.append(f"CUresult {name}({', '.join(param_strs)}) {{")
        lines.append(f"    // {func.get('comment', 'Handled locally on client side')}")
        body = func.get("local_body", "return CUDA_SUCCESS;")
        for line in body.split('\n'):
            lines.append(f"    {line}")
        lines.append("}")
        return '\n'.join(lines)

    def _gen_stub_shim(self, func: dict) -> str:
        """Generate a shim that just returns a fixed error code."""
        name = func["name"]
        ret = func.get("stub_return", 0)
        params = func.get("params", [])
        param_strs = [f"{p['type']} {p['name']}" for p in params]

        lines = [f"// {name} — stub (returns {ret})"]
        lines.append(f"CUresult {name}({', '.join(param_strs)}) {{")
        lines.append(f"    return static_cast<CUresult>({ret});")
        lines.append("}")
        return '\n'.join(lines)

    def _gen_rpc_shim(self, func: dict) -> str:
        """Generate a full RPC shim function."""
        name = func["name"]
        rpc_name = snake_case(name)
        params = func.get("params", [])
        category = func.get("category", "enqueued")
        idx = self.rpc_index[name]

        param_strs = [f"{p['type']} {p['name']}" for p in params]
        lines = [f"// {name} | {category}"]
        lines.append(f"CUresult {name}({', '.join(param_strs)}) {{")

        # 1. init_shim
        lines.append("    init_shim();")
        lines.append("    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;")

        # 2. Translate input handles
        for p in params:
            if p.get("direction", "in") in ("in", "inout"):
                if is_input_handle(p):
                    pname = p["name"]
                    lines.append(f"    auto server_{pname} = translate_handle(reinterpret_cast<uint64_t>({pname}));")
                    lines.append(f"    if (!server_{pname}) return CUDA_ERROR_INVALID_HANDLE;")
                elif is_devptr(p):
                    pname = p["name"]
                    lines.append(f"    auto remote_{pname} = g_ptr_map.to_remote({pname});")
                    lines.append(f"    if (!remote_{pname}) return CUDA_ERROR_INVALID_DEVICE_POINTER;")

        # 3. Host sync for relevant params
        for p in params:
            if is_host_ptr(p):
                pname = p["name"]
                size_param = p.get("host_sync_size", "")
                if size_param:
                    lines.append(f"    sync_host_to_server({pname}, {size_param});")
                else:
                    lines.append(f"    // TODO: sync_host_to_server({pname}, <size>);")

        # 4. Build RPC call arguments
        rpc_args = []
        for p in params:
            pname = p["name"]
            if p.get("direction", "in") in ("in", "inout"):
                if is_input_handle(p):
                    rpc_args.append(f"*server_{pname}")
                elif is_devptr(p):
                    rpc_args.append(f"*remote_{pname}")
                elif p["type"].strip() == "const char*":
                    rpc_args.append(f"std::string({pname})")
                elif p["type"].strip() in ("CUdevice",):
                    rpc_args.append(f"static_cast<int32_t>({pname})")
                elif p["type"].strip() in ("size_t",):
                    rpc_args.append(f"static_cast<uint64_t>({pname})")
                elif p["type"].strip() in ("unsigned int",):
                    rpc_args.append(f"static_cast<uint32_t>({pname})")
                elif p["type"].strip() in ("unsigned long long", "cuuint64_t"):
                    rpc_args.append(f"static_cast<uint64_t>({pname})")
                else:
                    rpc_args.append(pname)

        # 5. Make RPC call
        lines.append(f"    auto ret = cuda_rpc_call<func_index::{rpc_name}>({', '.join(rpc_args)});")

        # 6. Register produced handles
        for hp in func.get("handle_produces", []):
            handle_type = func.get("handle_type_map", {}).get(hp, "handle")
            lines.append(f"    if (ret.result == CUDA_SUCCESS && ret.{hp} != 0) {{")
            lines.append(f"        uint64_t client_handle = allocate_client_handle(\"{handle_type}\");")
            lines.append(f"        register_handle(client_handle, ret.{hp}, \"{handle_type}\");")

            # Find the output param that corresponds to this handle
            for p in params:
                if p.get("direction") == "out" and is_handle_type(p["type"].rstrip("*")):
                    lines.append(f"        *{p['name']} = reinterpret_cast<{p['type'].rstrip('*')}>(client_handle);")

            lines.append("    }")

        # 7. Register produced devptrs
        for dp in func.get("devptr_produces", []):
            lines.append(f"    if (ret.result == CUDA_SUCCESS) {{")
            lines.append(f"        CUdeviceptr client_ptr = g_ptr_map.map(ret.{dp});")
            # Find the output param
            for p in params:
                if p.get("direction") == "out" and p["type"].strip() == "CUdeviceptr*":
                    lines.append(f"        *{p['name']} = client_ptr;")
            lines.append("    }")

        # 8. Set output scalar values
        for p in params:
            if p.get("direction") == "out" and not is_handle_type(p["type"].rstrip("*")):
                pname = p["name"]
                ptype = p["type"].strip()
                if ptype == "CUdevice*":
                    lines.append(f"    if (ret.result == CUDA_SUCCESS) *{pname} = ret.{pname};")
                elif ptype in ("size_t*", "unsigned int*", "int*"):
                    lines.append(f"    if (ret.result == CUDA_SUCCESS) *{pname} = static_cast<{ptype.rstrip('*')}>(ret.{pname});")
                elif ptype == "float*":
                    lines.append(f"    if (ret.result == CUDA_SUCCESS) *{pname} = ret.{pname};")
                elif ptype == "unsigned long long*":
                    lines.append(f"    if (ret.result == CUDA_SUCCESS) *{pname} = static_cast<unsigned long long>(ret.{pname});")

        # 9. Readback from server
        for p in params:
            if needs_readback(p):
                pname = p["name"]
                size_param = p.get("readback_size", "")
                if size_param:
                    lines.append(f"    if (ret.result == CUDA_SUCCESS) readback_from_server({pname}, {size_param});")

        # 10. Return result
        lines.append("    return static_cast<CUresult>(ret.result);")
        lines.append("}")
        return '\n'.join(lines)

    # ── Generate gen_server.inc ───────────────────────────────────────
    def gen_server_inc(self) -> str:
        """Generate the server-side handler bodies."""
        lines = []
        lines.append("// AUTO-GENERATED by codegen/codegen.py — DO NOT EDIT")
        lines.append(f"// Contains {self.total_active()} CUDA Driver API server handler bodies")
        lines.append("// Included in cuda_server_codegen.cpp")
        lines.append("")

        for func in self.active_functions:
            if func.get("local_only", False) or func.get("stub_return") is not None:
                continue  # No server handler needed
            lines.append(self._gen_server_handler(func))
            lines.append("")

        return '\n'.join(lines)

    def _gen_server_handler(self, func: dict) -> str:
        """Generate a server-side handler for a CUDA function."""
        name = func["name"]
        rpc_name = snake_case(name)
        params = func.get("params", [])

        lines = [f"// ── {name} (server handler) ──"]

        # Build the handler function signature
        # Server handlers receive deserialized params and return the result struct
        struct_name = f"{pascal_case(rpc_name)}Ret"

        # Input params for the server handler
        handler_params = []
        for p in params:
            if p.get("direction", "in") in ("in", "inout"):
                ptype = cpp_rpc_type(p["type"])
                pname = p["name"]
                if p["type"].strip() == "const char*":
                    ptype = "std::string"
                handler_params.append(f"{ptype} {pname}")

        # Add handle produces as extra params? No, they're output.
        # But we need to pass them for translation

        lines.append(f"// {name} → real CUDA call on GPU")

        # Generate the actual handler implementation
        # This goes inside the zpp::bits server dispatch, so we generate
        # a standalone function that the server can call.

        lines.append(f"static {struct_name} handle_{rpc_name}({', '.join(handler_params)}) {{")

        # 1. Translate virtual handles to real handles
        for p in params:
            if p.get("direction", "in") in ("in", "inout"):
                if is_input_handle(p):
                    pname = p["name"]
                    real_type = p["type"].strip().replace("const ", "")
                    lines.append(f"    {real_type} real_{pname} = reinterpret_cast<{real_type}>(g_vhandles.translate({pname}));")
                elif is_devptr(p):
                    pname = p["name"]
                    lines.append(f"    CUdeviceptr real_{pname} = static_cast<CUdeviceptr>(g_vhandles.translate({pname}));")

        # 2. Prepare output variables
        for p in params:
            if p.get("direction") == "out":
                ptype = p["type"].strip().rstrip("*")
                pname = p["name"]
                if is_handle_type(ptype):
                    lines.append(f"    {ptype} real_{pname} = nullptr;")
                elif ptype == "CUdeviceptr":
                    lines.append(f"    CUdeviceptr real_{pname} = 0;")
                elif ptype in ("size_t", "unsigned int", "int", "float", "unsigned long long", "long long"):
                    lines.append(f"    {ptype} real_{pname} = 0;")
                elif ptype == "char":
                    lines.append(f"    char real_{pname}[256] = {{}};")

        # 3. Build the real CUDA call arguments
        call_args = []
        for p in params:
            pname = p["name"]
            if p.get("direction") == "out":
                if is_handle_type(p["type"].strip().rstrip("*")):
                    call_args.append(f"&real_{pname}")
                elif p["type"].strip() == "CUdeviceptr*":
                    call_args.append(f"&real_{pname}")
                elif p["type"].strip() in ("size_t*", "unsigned int*", "int*", "float*",
                                           "unsigned long long*", "long long*"):
                    call_args.append(f"&real_{pname}")
                else:
                    call_args.append(f"&real_{pname}")
            else:
                if is_input_handle(p):
                    call_args.append(f"real_{pname}")
                elif is_devptr(p):
                    call_args.append(f"real_{pname}")
                elif p["type"].strip() == "const char*":
                    call_args.append(f"{pname}.c_str()")
                elif p["type"].strip() == "CUdevice":
                    call_args.append(f"static_cast<CUdevice>({pname})")
                elif p["type"].strip() == "size_t":
                    call_args.append(f"static_cast<size_t>({pname})")
                elif p["type"].strip() == "unsigned int":
                    call_args.append(f"static_cast<unsigned int>({pname})")
                elif p["type"].strip() == "unsigned long long":
                    call_args.append(f"static_cast<unsigned long long>({pname})")
                else:
                    call_args.append(pname)

        # 4. Make the real CUDA call
        lines.append(f"    CUresult res = {name}({', '.join(call_args)});")

        # 5. Build the return struct
        lines.append(f"    {struct_name} ret;")
        lines.append(f"    ret.result = static_cast<int32_t>(res);")

        # 6. Fill in output values
        for p in params:
            if p.get("direction") == "out":
                pname = p["name"]
                ptype = p["type"].strip().rstrip("*")
                if is_handle_type(ptype):
                    lines.append(f"    ret.{pname} = reinterpret_cast<uint64_t>(real_{pname});")
                    # Register handle in server's virtual handle table
                    lines.append(f"    if (res == CUDA_SUCCESS && real_{pname} != nullptr) {{")
                    lines.append(f"        g_has_produced_handle = true;")
                    lines.append(f"    }}")
                elif ptype == "CUdeviceptr":
                    lines.append(f"    ret.{pname} = static_cast<uint64_t>(real_{pname});")
                    lines.append(f"    if (res == CUDA_SUCCESS && real_{pname} != 0) {{")
                    lines.append(f"        g_has_produced_handle = true;")
                    lines.append(f"    }}")
                elif ptype == "CUdevice":
                    lines.append(f"    ret.{pname} = static_cast<int32_t>(real_{pname});")
                elif ptype in ("size_t",):
                    lines.append(f"    ret.{pname} = static_cast<uint64_t>(real_{pname});")
                elif ptype in ("unsigned int",):
                    lines.append(f"    ret.{pname} = static_cast<uint32_t>(real_{pname});")
                elif ptype in ("int",):
                    lines.append(f"    ret.{pname} = static_cast<int32_t>(real_{pname});")
                elif ptype in ("float",):
                    lines.append(f"    ret.{pname} = real_{pname};")
                elif ptype in ("unsigned long long",):
                    lines.append(f"    ret.{pname} = static_cast<uint64_t>(real_{pname});")

        # Handle produces that are separate from output params
        for hp in func.get("handle_produces", []):
            # Already handled above via output params
            pass

        for dp in func.get("devptr_produces", []):
            # Already handled above via output params
            pass

        lines.append("    return ret;")
        lines.append("}")
        return '\n'.join(lines)

    # ── Generate get_proc_address symbol map ──────────────────────────
    def gen_symbol_map(self) -> str:
        """Generate the cuGetProcAddress symbol lookup table."""
        lines = []
        lines.append("// AUTO-GENERATED by codegen/codegen.py — DO NOT EDIT")
        lines.append("// Symbol names for cuGetProcAddress resolution")
        lines.append("")
        lines.append("static const char* g_cuda_symbol_names[] = {")
        for func in self.active_functions:
            name = func["name"]
            lines.append(f'    "{name}",')
        lines.append("    nullptr")
        lines.append("};")
        return '\n'.join(lines)

    # ── Main entry point ──────────────────────────────────────────────
    def generate_all(self, output_dir: str):
        """Generate all output files."""
        os.makedirs(output_dir, exist_ok=True)

        # gen_api.hpp
        api_hpp = self.gen_api_hpp()
        path = os.path.join(output_dir, "gen_api.hpp")
        with open(path, 'w') as f:
            f.write(api_hpp)
        print(f"Generated {path} ({self.total_active()} functions)")

        # gen_shim.inc
        shim_inc = self.gen_shim_inc()
        path = os.path.join(output_dir, "gen_shim.inc")
        with open(path, 'w') as f:
            f.write(shim_inc)
        print(f"Generated {path}")

        # gen_server.inc
        server_inc = self.gen_server_inc()
        path = os.path.join(output_dir, "gen_server.inc")
        with open(path, 'w') as f:
            f.write(server_inc)
        print(f"Generated {path}")

        # gen_symbol_map.inc
        symbol_map = self.gen_symbol_map()
        path = os.path.join(output_dir, "gen_symbol_map.inc")
        with open(path, 'w') as f:
            f.write(symbol_map)
        print(f"Generated {path}")

        # Stats
        total = len(self.functions)
        active = self.total_active()
        disabled = sum(1 for f in self.functions if f.get("disabled", False))
        local_only = sum(1 for f in self.active_functions if f.get("local_only", False))
        stubbed = sum(1 for f in self.active_functions if f.get("stub_return") is not None)
        rpc_funcs = active - local_only - stubbed
        print(f"\nStats: {total} total, {active} active, {disabled} disabled, "
              f"{local_only} local_only, {stubbed} stubbed, {rpc_funcs} RPC")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "cuda_functions.json")
    output_dir = os.path.join(script_dir)  # Output alongside codegen.py

    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    if not os.path.exists(db_path):
        print(f"Error: Function database not found at {db_path}")
        print("Run build_db.py first to generate the function database.")
        sys.exit(1)

    codegen = CudaCodegen(db_path)
    codegen.generate_all(output_dir)
