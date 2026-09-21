llvm.func @_cudaOccupancyMaxActiveBlocksPerMultiprocessor(!llvm.ptr, !llvm.ptr, i32, i64) -> i32

llvm.func @_cudaLaunchKernelEx(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

llvm.func @_cudaLaunchKernel(!llvm.ptr, i32, i32, i32, i32, i32, i32, !llvm.ptr, i64, !llvm.ptr) -> i32

llvm.func @_cudaLibraryLoadData(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32) -> i32

llvm.func @_cuKernelGetAttribute(!llvm.ptr, i32, !llvm.ptr, i32) -> i32

llvm.func @_cudaDeviceGetAttribute(!llvm.ptr, i32, i32) -> i32

llvm.func @_cudaGetDevice(!llvm.ptr) -> i32

llvm.func @_cudaKernelSetAttributeForDevice(!llvm.ptr, i32, i32, i32) -> i32

llvm.func @_cudaFuncSetAttribute(!llvm.ptr, i32, i32) -> i32

llvm.func @_cudaLibraryGetKernel(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

llvm.func @printf(!llvm.ptr, ...) -> i32

llvm.func @cuda_num_binaries() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %0 : i32
}

llvm.func @cuda_init(%arg0: !llvm.ptr) -> i32 {
  %0 = llvm.mlir.zero : !llvm.ptr
  %1 = llvm.mlir.addressof @kernels_binary : !llvm.ptr
  %2 = llvm.mlir.zero : i32
  %3 = llvm.getelementptr %arg0[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
  %4 = llvm.call @_cudaLibraryLoadData(%3, %1, %0, %0, %2, %0, %0, %2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32) -> i32
  %5 = llvm.mlir.constant(0 : i32) : i32
  %6 = llvm.icmp "eq" %4, %5 : i32
  llvm.cond_br %6, ^bb1, ^bb2(%4 : i32)
^bb1:  // pred: ^bb0
  %7 = llvm.mlir.constant(0 : i32) : i32
  llvm.br ^bb2(%7 : i32)
^bb2(%8: i32):  // 2 preds: ^bb0, ^bb1
  llvm.return %8 : i32
}

llvm.func @cuda_load(%arg0: !llvm.ptr) -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  %2 = llvm.call @_cudaGetDevice(%1) : (!llvm.ptr) -> i32
  %3 = llvm.mlir.constant(0 : i32) : i32
  %4 = llvm.icmp "eq" %2, %3 : i32
  llvm.cond_br %4, ^bb1, ^bb7(%2 : i32)
^bb1:  // pred: ^bb0
  %5 = llvm.load %1 : !llvm.ptr -> i32
  %6 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  %7 = llvm.mlir.constant(97 : i32) : i32
  %8 = llvm.call @_cudaDeviceGetAttribute(%6, %7, %5) : (!llvm.ptr, i32, i32) -> i32
  %9 = llvm.mlir.constant(0 : i32) : i32
  %10 = llvm.icmp "eq" %8, %9 : i32
  llvm.cond_br %10, ^bb2, ^bb7(%8 : i32)
^bb2:  // pred: ^bb1
  %11 = llvm.load %6 : !llvm.ptr -> i32
  %12 = llvm.mlir.addressof @kernels_kernel_cutlass_kernel__upstream_blackwell_gemmDenseGemmKernel_object_at__TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom_ThrID10_ShapeMNK12812816_TVLayoutA1128161281128_TVLayoutB_0 : !llvm.ptr
  %13 = llvm.mlir.constant("kernel_cutlass_kernel__upstream_blackwell_gemmDenseGemmKernel_object_at__TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom_ThrID10_ShapeMNK12812816_TVLayoutA1128161281128_TVLayoutB_0\00") : !llvm.array<190 x i8>
  %14 = llvm.mlir.constant(1 : i32) : i32
  %15 = llvm.alloca %14 x !llvm.array<190 x i8> : (i32) -> !llvm.ptr
  llvm.store %13, %15 : !llvm.array<190 x i8>, !llvm.ptr
  %16 = llvm.getelementptr %arg0[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
  %17 = llvm.load %16 : !llvm.ptr -> !llvm.ptr
  %18 = llvm.call @_cudaLibraryGetKernel(%12, %17, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  %19 = llvm.mlir.constant(0 : i32) : i32
  %20 = llvm.icmp "eq" %18, %19 : i32
  llvm.cond_br %20, ^bb3, ^bb7(%18 : i32)
^bb3:  // pred: ^bb2
  %21 = llvm.load %12 : !llvm.ptr -> !llvm.ptr
  %22 = llvm.mlir.constant(8 : i32) : i32
  %23 = llvm.mlir.constant(1 : i32) : i32
  %24 = llvm.mlir.constant(1 : i32) : i32
  %25 = llvm.alloca %23 x i32 : (i32) -> !llvm.ptr
  %26 = llvm.call @_cuKernelGetAttribute(%25, %24, %21, %5) : (!llvm.ptr, i32, !llvm.ptr, i32) -> i32
  %27 = llvm.mlir.constant(0 : i32) : i32
  %28 = llvm.icmp "eq" %26, %27 : i32
  llvm.cond_br %28, ^bb4, ^bb7(%26 : i32)
^bb4:  // pred: ^bb3
  %29 = llvm.load %25 : !llvm.ptr -> i32
  %30 = llvm.sub %11, %29 : i32
  %31 = llvm.call @_cudaFuncSetAttribute(%21, %22, %30) : (!llvm.ptr, i32, i32) -> i32
  %32 = llvm.mlir.constant(0 : i32) : i32
  %33 = llvm.icmp "eq" %31, %32 : i32
  llvm.cond_br %33, ^bb5, ^bb7(%31 : i32)
^bb5:  // pred: ^bb4
  %34 = llvm.mlir.constant(14 : i32) : i32
  %35 = llvm.mlir.constant(1 : i32) : i32
  %36 = llvm.call @_cudaFuncSetAttribute(%21, %34, %35) : (!llvm.ptr, i32, i32) -> i32
  %37 = llvm.mlir.constant(0 : i32) : i32
  %38 = llvm.icmp "eq" %36, %37 : i32
  llvm.cond_br %38, ^bb6, ^bb7(%36 : i32)
^bb6:  // pred: ^bb5
  %39 = llvm.mlir.constant(0 : i32) : i32
  llvm.br ^bb7(%39 : i32)
^bb7(%40: i32):  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
  llvm.return %40 : i32
}

llvm.func @cuda_load_to_device(%arg0: !llvm.ptr, %arg1: i32) -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  %2 = llvm.mlir.constant(97 : i32) : i32
  %3 = llvm.call @_cudaDeviceGetAttribute(%1, %2, %arg1) : (!llvm.ptr, i32, i32) -> i32
  %4 = llvm.mlir.constant(0 : i32) : i32
  %5 = llvm.icmp "eq" %3, %4 : i32
  llvm.cond_br %5, ^bb1, ^bb6(%3 : i32)
^bb1:  // pred: ^bb0
  %6 = llvm.load %1 : !llvm.ptr -> i32
  %7 = llvm.mlir.addressof @kernels_kernel_cutlass_kernel__upstream_blackwell_gemmDenseGemmKernel_object_at__TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom_ThrID10_ShapeMNK12812816_TVLayoutA1128161281128_TVLayoutB_0 : !llvm.ptr
  %8 = llvm.mlir.constant("kernel_cutlass_kernel__upstream_blackwell_gemmDenseGemmKernel_object_at__TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom_ThrID10_ShapeMNK12812816_TVLayoutA1128161281128_TVLayoutB_0\00") : !llvm.array<190 x i8>
  %9 = llvm.mlir.constant(1 : i32) : i32
  %10 = llvm.alloca %9 x !llvm.array<190 x i8> : (i32) -> !llvm.ptr
  llvm.store %8, %10 : !llvm.array<190 x i8>, !llvm.ptr
  %11 = llvm.getelementptr %arg0[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
  %12 = llvm.load %11 : !llvm.ptr -> !llvm.ptr
  %13 = llvm.call @_cudaLibraryGetKernel(%7, %12, %10) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  %14 = llvm.mlir.constant(0 : i32) : i32
  %15 = llvm.icmp "eq" %13, %14 : i32
  llvm.cond_br %15, ^bb2, ^bb6(%13 : i32)
^bb2:  // pred: ^bb1
  %16 = llvm.load %7 : !llvm.ptr -> !llvm.ptr
  %17 = llvm.mlir.constant(8 : i32) : i32
  %18 = llvm.mlir.constant(1 : i32) : i32
  %19 = llvm.mlir.constant(1 : i32) : i32
  %20 = llvm.alloca %18 x i32 : (i32) -> !llvm.ptr
  %21 = llvm.call @_cuKernelGetAttribute(%20, %19, %16, %arg1) : (!llvm.ptr, i32, !llvm.ptr, i32) -> i32
  %22 = llvm.mlir.constant(0 : i32) : i32
  %23 = llvm.icmp "eq" %21, %22 : i32
  llvm.cond_br %23, ^bb3, ^bb6(%21 : i32)
^bb3:  // pred: ^bb2
  %24 = llvm.load %20 : !llvm.ptr -> i32
  %25 = llvm.sub %6, %24 : i32
  %26 = llvm.call @_cudaKernelSetAttributeForDevice(%16, %17, %25, %arg1) : (!llvm.ptr, i32, i32, i32) -> i32
  %27 = llvm.mlir.constant(0 : i32) : i32
  %28 = llvm.icmp "eq" %26, %27 : i32
  llvm.cond_br %28, ^bb4, ^bb6(%26 : i32)
^bb4:  // pred: ^bb3
  %29 = llvm.mlir.constant(14 : i32) : i32
  %30 = llvm.mlir.constant(1 : i32) : i32
  %31 = llvm.call @_cudaKernelSetAttributeForDevice(%16, %29, %30, %arg1) : (!llvm.ptr, i32, i32, i32) -> i32
  %32 = llvm.mlir.constant(0 : i32) : i32
  %33 = llvm.icmp "eq" %31, %32 : i32
  llvm.cond_br %33, ^bb5, ^bb6(%31 : i32)
^bb5:  // pred: ^bb4
  %34 = llvm.mlir.constant(0 : i32) : i32
  llvm.br ^bb6(%34 : i32)
^bb6(%35: i32):  // 6 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5
  llvm.return %35 : i32
}

llvm.func @cutlass___call____upstream_blackwell_gemmDenseGemmKernel_object_at__Tensorgmemoi641i64_Tensorgmemoi641i64_Tensorgmemoi641i64_CUstream0xc44a5a0_functionDenseGemmKernellambdaat(%arg0: !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)>, %arg1: !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)>, %arg2: !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)>, %arg3: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface} {
  %0 = llvm.mlir.constant(2 : i32) : i32
  %1 = llvm.mlir.constant(4 : i32) : i32
  %2 = llvm.mlir.constant(17 : i32) : i32
  %3 = llvm.mlir.addressof @"%s\0A" : !llvm.ptr
  %4 = llvm.mlir.addressof @"ERROR: Reached max number of attributes, unable to add more attributes." : !llvm.ptr
  %5 = llvm.mlir.constant(6 : i32) : i32
  %6 = llvm.mlir.poison : !llvm.struct<(struct<()>, struct<(struct<(i32, i32, i32)>, struct<()>)>)>
  %7 = llvm.mlir.poison : !llvm.struct<(struct<(i32, i32, i32)>, struct<()>)>
  %8 = llvm.mlir.poison : !llvm.struct<(struct<(array<16 x i64>)>)>
  %9 = llvm.mlir.poison : !llvm.struct<(i1, i1, i1, vector<4xi32>)>
  %10 = llvm.mlir.constant(2233785415175766016 : i64) : i64
  %11 = llvm.mlir.constant(287650 : i64) : i64
  %12 = llvm.mlir.constant(127 : i64) : i64
  %13 = llvm.mlir.constant(4539628424389459968 : i64) : i64
  %14 = llvm.mlir.constant(287522 : i64) : i64
  %15 = llvm.mlir.constant(dense<0> : vector<4xi32>) : vector<4xi32>
  %16 = llvm.mlir.constant(1 : i32) : i32
  %17 = llvm.mlir.constant(196736 : i64) : i64
  %18 = llvm.mlir.poison : !llvm.struct<()>
  %19 = llvm.mlir.constant(128 : i32) : i32
  %20 = llvm.mlir.constant(15 : i64) : i64
  %21 = llvm.mlir.constant(36 : i64) : i64
  %22 = llvm.mlir.constant(21 : i64) : i64
  %23 = llvm.mlir.constant(131072 : i64) : i64
  %24 = llvm.mlir.constant(8 : i64) : i64
  %25 = llvm.mlir.constant(32 : i64) : i64
  %26 = llvm.mlir.constant(9007199254740991 : i64) : i64
  %27 = llvm.mlir.constant(16 : i64) : i64
  %28 = llvm.mlir.constant(4 : i64) : i64
  %29 = llvm.mlir.constant(4294967295 : i64) : i64
  %30 = llvm.mlir.constant(2 : i64) : i64
  %31 = llvm.mlir.constant(1 : i64) : i64
  %32 = llvm.mlir.constant(0 : i64) : i64
  %33 = llvm.mlir.constant(16 : i32) : i32
  %34 = llvm.mlir.constant(0 : i32) : i32
  %35 = llvm.mlir.constant(false) : i1
  %36 = llvm.insertvalue %35, %9[0] : !llvm.struct<(i1, i1, i1, vector<4xi32>)> 
  %37 = llvm.insertvalue %35, %36[1] : !llvm.struct<(i1, i1, i1, vector<4xi32>)> 
  %38 = llvm.insertvalue %35, %37[2] : !llvm.struct<(i1, i1, i1, vector<4xi32>)> 
  %39 = llvm.insertvalue %15, %38[3] : !llvm.struct<(i1, i1, i1, vector<4xi32>)> 
  %40 = llvm.alloca %33 x i64 {alignment = 64 : i64} : (i32) -> !llvm.ptr
  %41 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %42 = llvm.extractvalue %arg0[1, 0, 0] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %43 = llvm.extractvalue %arg0[1, 0, 1] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %44 = llvm.extractvalue %arg0[1, 0, 2] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %45 = llvm.extractvalue %arg0[1, 1, 0] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %46 = llvm.extractvalue %arg0[1, 1, 1] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %47 = llvm.sext %43 : i32 to i64
  %48 = llvm.sext %42 : i32 to i64
  %49 = llvm.mul %45, %30 : i64
  %50 = llvm.sext %44 : i32 to i64
  %51 = llvm.mul %46, %30 : i64
  %52 = llvm.ptrtoint %41 : !llvm.ptr<1> to i64
  %53 = llvm.getelementptr %40[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %53 : i64, !llvm.ptr
  %54 = llvm.getelementptr %40[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %54 : i64, !llvm.ptr
  %55 = llvm.getelementptr %40[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %55 : i64, !llvm.ptr
  %56 = llvm.getelementptr %40[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %56 : i64, !llvm.ptr
  %57 = llvm.getelementptr %40[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %57 : i64, !llvm.ptr
  %58 = llvm.getelementptr %40[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %58 : i64, !llvm.ptr
  %59 = llvm.getelementptr %40[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %59 : i64, !llvm.ptr
  %60 = llvm.getelementptr %40[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %60 : i64, !llvm.ptr
  %61 = llvm.udiv %52, %27 : i64
  %62 = llvm.and %61, %26 : i64
  %63 = llvm.shl %62, %28 : i64
  llvm.store %63, %53 : i64, !llvm.ptr
  %64 = llvm.sub %48, %31 : i64
  %65 = llvm.sub %50, %31 : i64
  %66 = llvm.mul %64, %49 : i64
  %67 = llvm.mul %65, %51 : i64
  %68 = llvm.add %66, %67 : i64
  %69 = llvm.mul %47, %27 : i64
  %70 = llvm.udiv %69, %24 : i64
  %71 = llvm.add %70, %68 : i64
  %72 = llvm.icmp "uge" %71, %23 : i64
  %73 = llvm.zext %72 : i1 to i64
  %74 = llvm.shl %73, %22 : i64
  %75 = llvm.udiv %49, %27 : i64
  %76 = llvm.shl %75, %25 : i64
  %77 = llvm.or %74, %76 : i64
  %78 = llvm.or %77, %14 : i64
  llvm.store %78, %54 : i64, !llvm.ptr
  %79 = llvm.udiv %51, %27 : i64
  %80 = llvm.and %79, %29 : i64
  llvm.store %80, %55 : i64, !llvm.ptr
  %81 = llvm.lshr %49, %21 : i64
  %82 = llvm.and %81, %20 : i64
  %83 = llvm.shl %82, %25 : i64
  %84 = llvm.lshr %51, %21 : i64
  %85 = llvm.and %84, %20 : i64
  %86 = llvm.shl %85, %21 : i64
  %87 = llvm.or %83, %86 : i64
  llvm.store %87, %56 : i64, !llvm.ptr
  %88 = llvm.sub %47, %31 : i64
  %89 = llvm.and %88, %29 : i64
  %90 = llvm.shl %64, %25 : i64
  %91 = llvm.or %89, %90 : i64
  llvm.store %91, %57 : i64, !llvm.ptr
  %92 = llvm.and %65, %29 : i64
  llvm.store %92, %58 : i64, !llvm.ptr
  llvm.store %13, %59 : i64, !llvm.ptr
  llvm.store %12, %60 : i64, !llvm.ptr
  %93 = llvm.ptrtoint %40 : !llvm.ptr to i64
  %94 = llvm.inttoptr %93 : i64 to !llvm.ptr
  %95 = llvm.load %94 {nontemporal} : !llvm.ptr -> !llvm.struct<(array<16 x i64>)>
  %96 = llvm.insertvalue %95, %8[0] : !llvm.struct<(struct<(array<16 x i64>)>)> 
  %97 = llvm.extractvalue %arg0[1, 0] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %98 = llvm.insertvalue %97, %7[0] : !llvm.struct<(struct<(i32, i32, i32)>, struct<()>)> 
  %99 = llvm.insertvalue %18, %98[1] : !llvm.struct<(struct<(i32, i32, i32)>, struct<()>)> 
  %100 = llvm.insertvalue %18, %6[0] : !llvm.struct<(struct<()>, struct<(struct<(i32, i32, i32)>, struct<()>)>)> 
  %101 = llvm.insertvalue %99, %100[1] : !llvm.struct<(struct<()>, struct<(struct<(i32, i32, i32)>, struct<()>)>)> 
  %102 = llvm.alloca %33 x i64 {alignment = 64 : i64} : (i32) -> !llvm.ptr
  %103 = llvm.extractvalue %arg1[0] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %104 = llvm.extractvalue %arg1[1, 0, 0] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %105 = llvm.extractvalue %arg1[1, 0, 1] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %106 = llvm.extractvalue %arg1[1, 0, 2] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %107 = llvm.extractvalue %arg1[1, 1, 0] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %108 = llvm.extractvalue %arg1[1, 1, 1] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %109 = llvm.sext %105 : i32 to i64
  %110 = llvm.sext %104 : i32 to i64
  %111 = llvm.mul %107, %30 : i64
  %112 = llvm.sext %106 : i32 to i64
  %113 = llvm.mul %108, %30 : i64
  %114 = llvm.ptrtoint %103 : !llvm.ptr<1> to i64
  %115 = llvm.getelementptr %102[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %115 : i64, !llvm.ptr
  %116 = llvm.getelementptr %102[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %116 : i64, !llvm.ptr
  %117 = llvm.getelementptr %102[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %117 : i64, !llvm.ptr
  %118 = llvm.getelementptr %102[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %118 : i64, !llvm.ptr
  %119 = llvm.getelementptr %102[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %119 : i64, !llvm.ptr
  %120 = llvm.getelementptr %102[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %120 : i64, !llvm.ptr
  %121 = llvm.getelementptr %102[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %121 : i64, !llvm.ptr
  %122 = llvm.getelementptr %102[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %122 : i64, !llvm.ptr
  %123 = llvm.udiv %114, %27 : i64
  %124 = llvm.and %123, %26 : i64
  %125 = llvm.shl %124, %28 : i64
  llvm.store %125, %115 : i64, !llvm.ptr
  %126 = llvm.sub %110, %31 : i64
  %127 = llvm.sub %112, %31 : i64
  %128 = llvm.mul %126, %111 : i64
  %129 = llvm.mul %127, %113 : i64
  %130 = llvm.add %128, %129 : i64
  %131 = llvm.mul %109, %27 : i64
  %132 = llvm.udiv %131, %24 : i64
  %133 = llvm.add %132, %130 : i64
  %134 = llvm.icmp "uge" %133, %23 : i64
  %135 = llvm.zext %134 : i1 to i64
  %136 = llvm.shl %135, %22 : i64
  %137 = llvm.udiv %111, %27 : i64
  %138 = llvm.shl %137, %25 : i64
  %139 = llvm.or %136, %138 : i64
  %140 = llvm.or %139, %14 : i64
  llvm.store %140, %116 : i64, !llvm.ptr
  %141 = llvm.udiv %113, %27 : i64
  %142 = llvm.and %141, %29 : i64
  llvm.store %142, %117 : i64, !llvm.ptr
  %143 = llvm.lshr %111, %21 : i64
  %144 = llvm.and %143, %20 : i64
  %145 = llvm.shl %144, %25 : i64
  %146 = llvm.lshr %113, %21 : i64
  %147 = llvm.and %146, %20 : i64
  %148 = llvm.shl %147, %21 : i64
  %149 = llvm.or %145, %148 : i64
  llvm.store %149, %118 : i64, !llvm.ptr
  %150 = llvm.sub %109, %31 : i64
  %151 = llvm.and %150, %29 : i64
  %152 = llvm.shl %126, %25 : i64
  %153 = llvm.or %151, %152 : i64
  llvm.store %153, %119 : i64, !llvm.ptr
  %154 = llvm.and %127, %29 : i64
  llvm.store %154, %120 : i64, !llvm.ptr
  llvm.store %13, %121 : i64, !llvm.ptr
  llvm.store %12, %122 : i64, !llvm.ptr
  %155 = llvm.ptrtoint %102 : !llvm.ptr to i64
  %156 = llvm.inttoptr %155 : i64 to !llvm.ptr
  %157 = llvm.load %156 {nontemporal} : !llvm.ptr -> !llvm.struct<(array<16 x i64>)>
  %158 = llvm.insertvalue %157, %8[0] : !llvm.struct<(struct<(array<16 x i64>)>)> 
  %159 = llvm.extractvalue %arg1[1, 0] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %160 = llvm.insertvalue %159, %7[0] : !llvm.struct<(struct<(i32, i32, i32)>, struct<()>)> 
  %161 = llvm.insertvalue %18, %160[1] : !llvm.struct<(struct<(i32, i32, i32)>, struct<()>)> 
  %162 = llvm.insertvalue %161, %100[1] : !llvm.struct<(struct<()>, struct<(struct<(i32, i32, i32)>, struct<()>)>)> 
  %163 = llvm.alloca %33 x i64 {alignment = 64 : i64} : (i32) -> !llvm.ptr
  %164 = llvm.extractvalue %arg2[0] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %165 = llvm.extractvalue %arg2[1, 0, 0] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %166 = llvm.extractvalue %arg2[1, 0, 1] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %167 = llvm.extractvalue %arg2[1, 0, 2] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %168 = llvm.extractvalue %arg2[1, 1, 0] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %169 = llvm.extractvalue %arg2[1, 1, 1] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %170 = llvm.sext %166 : i32 to i64
  %171 = llvm.sext %165 : i32 to i64
  %172 = llvm.mul %168, %28 : i64
  %173 = llvm.sext %167 : i32 to i64
  %174 = llvm.mul %169, %28 : i64
  %175 = llvm.ptrtoint %164 : !llvm.ptr<1> to i64
  %176 = llvm.getelementptr %163[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %176 : i64, !llvm.ptr
  %177 = llvm.getelementptr %163[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %177 : i64, !llvm.ptr
  %178 = llvm.getelementptr %163[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %178 : i64, !llvm.ptr
  %179 = llvm.getelementptr %163[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %179 : i64, !llvm.ptr
  %180 = llvm.getelementptr %163[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %180 : i64, !llvm.ptr
  %181 = llvm.getelementptr %163[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %181 : i64, !llvm.ptr
  %182 = llvm.getelementptr %163[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %182 : i64, !llvm.ptr
  %183 = llvm.getelementptr %163[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %32, %183 : i64, !llvm.ptr
  %184 = llvm.udiv %175, %27 : i64
  %185 = llvm.and %184, %26 : i64
  %186 = llvm.shl %185, %28 : i64
  llvm.store %186, %176 : i64, !llvm.ptr
  %187 = llvm.sub %171, %31 : i64
  %188 = llvm.sub %173, %31 : i64
  %189 = llvm.mul %187, %172 : i64
  %190 = llvm.mul %188, %174 : i64
  %191 = llvm.add %189, %190 : i64
  %192 = llvm.mul %170, %25 : i64
  %193 = llvm.udiv %192, %24 : i64
  %194 = llvm.add %193, %191 : i64
  %195 = llvm.icmp "uge" %194, %23 : i64
  %196 = llvm.zext %195 : i1 to i64
  %197 = llvm.shl %196, %22 : i64
  %198 = llvm.udiv %172, %27 : i64
  %199 = llvm.shl %198, %25 : i64
  %200 = llvm.or %197, %199 : i64
  %201 = llvm.or %200, %11 : i64
  llvm.store %201, %177 : i64, !llvm.ptr
  %202 = llvm.udiv %174, %27 : i64
  %203 = llvm.and %202, %29 : i64
  llvm.store %203, %178 : i64, !llvm.ptr
  %204 = llvm.lshr %172, %21 : i64
  %205 = llvm.and %204, %20 : i64
  %206 = llvm.shl %205, %25 : i64
  %207 = llvm.lshr %174, %21 : i64
  %208 = llvm.and %207, %20 : i64
  %209 = llvm.shl %208, %21 : i64
  %210 = llvm.or %206, %209 : i64
  llvm.store %210, %179 : i64, !llvm.ptr
  %211 = llvm.sub %170, %31 : i64
  %212 = llvm.and %211, %29 : i64
  %213 = llvm.shl %187, %25 : i64
  %214 = llvm.or %212, %213 : i64
  llvm.store %214, %180 : i64, !llvm.ptr
  %215 = llvm.and %188, %29 : i64
  llvm.store %215, %181 : i64, !llvm.ptr
  llvm.store %10, %182 : i64, !llvm.ptr
  llvm.store %12, %183 : i64, !llvm.ptr
  %216 = llvm.ptrtoint %163 : !llvm.ptr to i64
  %217 = llvm.inttoptr %216 : i64 to !llvm.ptr
  %218 = llvm.load %217 {nontemporal} : !llvm.ptr -> !llvm.struct<(array<16 x i64>)>
  %219 = llvm.insertvalue %218, %8[0] : !llvm.struct<(struct<(array<16 x i64>)>)> 
  %220 = llvm.extractvalue %arg2[1, 0] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)> 
  %221 = llvm.insertvalue %220, %7[0] : !llvm.struct<(struct<(i32, i32, i32)>, struct<()>)> 
  %222 = llvm.insertvalue %18, %221[1] : !llvm.struct<(struct<(i32, i32, i32)>, struct<()>)> 
  %223 = llvm.insertvalue %222, %100[1] : !llvm.struct<(struct<()>, struct<(struct<(i32, i32, i32)>, struct<()>)>)> 
  %224 = llvm.sdiv %165, %19 : i32
  %225 = llvm.mul %224, %19 : i32
  %226 = llvm.icmp "ne" %165, %225 : i32
  %227 = llvm.icmp "slt" %165, %34 : i32
  %228 = llvm.icmp "eq" %227, %35 : i1
  %229 = llvm.and %226, %228 : i1
  %230 = llvm.add %224, %16 : i32
  %231 = llvm.select %229, %230, %224 : i1, i32
  %232 = llvm.sdiv %166, %19 : i32
  %233 = llvm.mul %232, %19 : i32
  %234 = llvm.icmp "ne" %166, %233 : i32
  %235 = llvm.icmp "slt" %166, %34 : i32
  %236 = llvm.icmp "eq" %235, %35 : i1
  %237 = llvm.and %234, %236 : i1
  %238 = llvm.add %232, %16 : i32
  %239 = llvm.select %237, %238, %232 : i1, i32
  %240 = llvm.alloca %16 x !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)> : (i32) -> !llvm.ptr
  %241 = llvm.alloca %16 x !llvm.array<17 x struct<(i32, array<4 x i8>, array<64 x i8>)>> : (i32) -> !llvm.ptr
  %242 = llvm.getelementptr %240[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  llvm.store %241, %242 : !llvm.ptr, !llvm.ptr
  %243 = llvm.getelementptr %240[0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  llvm.store %19, %243 : i32, !llvm.ptr
  %244 = llvm.getelementptr %240[0, 1, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  llvm.store %16, %244 : i32, !llvm.ptr
  %245 = llvm.getelementptr %240[0, 1, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  llvm.store %16, %245 : i32, !llvm.ptr
  %246 = llvm.getelementptr %240[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  llvm.store %17, %246 : i64, !llvm.ptr
  %247 = llvm.getelementptr %240[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  llvm.store %231, %247 : i32, !llvm.ptr
  %248 = llvm.getelementptr %240[0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  llvm.store %239, %248 : i32, !llvm.ptr
  %249 = llvm.getelementptr %240[0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  llvm.store %167, %249 : i32, !llvm.ptr
  %250 = llvm.getelementptr %240[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  llvm.store %34, %250 : i32, !llvm.ptr
  %251 = llvm.getelementptr %240[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  llvm.store %arg3, %251 : !llvm.ptr, !llvm.ptr
  %252 = llvm.alloca %16 x !llvm.array<1 x ptr> : (i32) -> !llvm.ptr
  %253 = llvm.getelementptr %252[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
  llvm.store %240, %253 : !llvm.ptr, !llvm.ptr
  %254 = llvm.load %253 : !llvm.ptr -> !llvm.ptr
  %255 = llvm.getelementptr %254[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  %256 = llvm.load %255 : !llvm.ptr -> i32
  %257 = llvm.getelementptr %254[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  %258 = llvm.load %257 : !llvm.ptr -> !llvm.ptr
  llvm.br ^bb7(%34 : i32)
^bb1(%259: i32):  // 2 preds: ^bb3, ^bb5
  %260 = llvm.getelementptr %258[%259] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  %261 = llvm.getelementptr %260[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  llvm.store %5, %261 : i32, !llvm.ptr
  %262 = llvm.getelementptr %260[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  llvm.store %34, %262 : i32, !llvm.ptr
  llvm.br ^bb8
^bb2:  // pred: ^bb4
  %263 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<72 x i8>
  %264 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
  %265 = llvm.call @printf(%264, %263) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
  llvm.intr.trap
  llvm.unreachable
^bb3:  // pred: ^bb4
  %266 = llvm.add %256, %16 : i32
  llvm.store %266, %255 : i32, !llvm.ptr
  llvm.br ^bb1(%256 : i32)
^bb4:  // pred: ^bb7
  %267 = llvm.icmp "uge" %256, %2 : i32
  llvm.cond_br %267, ^bb2, ^bb3
^bb5:  // pred: ^bb6
  llvm.br ^bb1(%273 : i32)
^bb6:  // pred: ^bb7
  %268 = llvm.getelementptr %258[%273] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  %269 = llvm.getelementptr %268[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  %270 = llvm.load %269 : !llvm.ptr -> i32
  %271 = llvm.icmp "eq" %270, %5 : i32
  %272 = llvm.add %273, %16 : i32
  llvm.cond_br %271, ^bb5, ^bb7(%272 : i32)
^bb7(%273: i32):  // 2 preds: ^bb0, ^bb6
  %274 = llvm.icmp "uge" %273, %256 : i32
  llvm.cond_br %274, ^bb4, ^bb6
^bb8:  // pred: ^bb1
  %275 = llvm.load %253 : !llvm.ptr -> !llvm.ptr
  %276 = llvm.getelementptr %275[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  %277 = llvm.load %276 : !llvm.ptr -> i32
  %278 = llvm.getelementptr %275[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  %279 = llvm.load %278 : !llvm.ptr -> !llvm.ptr
  llvm.br ^bb15(%34 : i32)
^bb9(%280: i32):  // 2 preds: ^bb11, ^bb13
  %281 = llvm.getelementptr %279[%280] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  %282 = llvm.getelementptr %281[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  llvm.store %1, %282 : i32, !llvm.ptr
  %283 = llvm.getelementptr %281[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  %284 = llvm.getelementptr %283[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32)>
  llvm.store %16, %284 : i32, !llvm.ptr
  %285 = llvm.getelementptr %283[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32)>
  llvm.store %16, %285 : i32, !llvm.ptr
  %286 = llvm.getelementptr %283[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32)>
  llvm.store %16, %286 : i32, !llvm.ptr
  llvm.br ^bb16
^bb10:  // pred: ^bb12
  %287 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<72 x i8>
  %288 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
  %289 = llvm.call @printf(%288, %287) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
  llvm.intr.trap
  llvm.unreachable
^bb11:  // pred: ^bb12
  %290 = llvm.add %277, %16 : i32
  llvm.store %290, %276 : i32, !llvm.ptr
  llvm.br ^bb9(%277 : i32)
^bb12:  // pred: ^bb15
  %291 = llvm.icmp "uge" %277, %2 : i32
  llvm.cond_br %291, ^bb10, ^bb11
^bb13:  // pred: ^bb14
  llvm.br ^bb9(%297 : i32)
^bb14:  // pred: ^bb15
  %292 = llvm.getelementptr %279[%297] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  %293 = llvm.getelementptr %292[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  %294 = llvm.load %293 : !llvm.ptr -> i32
  %295 = llvm.icmp "eq" %294, %1 : i32
  %296 = llvm.add %297, %16 : i32
  llvm.cond_br %295, ^bb13, ^bb15(%296 : i32)
^bb15(%297: i32):  // 2 preds: ^bb8, ^bb14
  %298 = llvm.icmp "uge" %297, %277 : i32
  llvm.cond_br %298, ^bb12, ^bb14
^bb16:  // pred: ^bb9
  %299 = llvm.load %253 : !llvm.ptr -> !llvm.ptr
  %300 = llvm.getelementptr %299[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  %301 = llvm.load %300 : !llvm.ptr -> i32
  %302 = llvm.getelementptr %299[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>
  %303 = llvm.load %302 : !llvm.ptr -> !llvm.ptr
  llvm.br ^bb23(%34 : i32)
^bb17(%304: i32):  // 2 preds: ^bb19, ^bb21
  %305 = llvm.getelementptr %303[%304] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  %306 = llvm.getelementptr %305[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  llvm.store %0, %306 : i32, !llvm.ptr
  %307 = llvm.getelementptr %305[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  llvm.store %34, %307 : i32, !llvm.ptr
  llvm.br ^bb24
^bb18:  // pred: ^bb20
  %308 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<72 x i8>
  %309 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
  %310 = llvm.call @printf(%309, %308) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
  llvm.intr.trap
  llvm.unreachable
^bb19:  // pred: ^bb20
  %311 = llvm.add %301, %16 : i32
  llvm.store %311, %300 : i32, !llvm.ptr
  llvm.br ^bb17(%301 : i32)
^bb20:  // pred: ^bb23
  %312 = llvm.icmp "uge" %301, %2 : i32
  llvm.cond_br %312, ^bb18, ^bb19
^bb21:  // pred: ^bb22
  llvm.br ^bb17(%318 : i32)
^bb22:  // pred: ^bb23
  %313 = llvm.getelementptr %303[%318] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  %314 = llvm.getelementptr %313[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, array<64 x i8>)>
  %315 = llvm.load %314 : !llvm.ptr -> i32
  %316 = llvm.icmp "eq" %315, %0 : i32
  %317 = llvm.add %318, %16 : i32
  llvm.cond_br %316, ^bb21, ^bb23(%317 : i32)
^bb23(%318: i32):  // 2 preds: ^bb16, ^bb22
  %319 = llvm.icmp "uge" %318, %301 : i32
  llvm.cond_br %319, ^bb20, ^bb22
^bb24:  // pred: ^bb17
  %320 = llvm.getelementptr %252[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
  %321 = llvm.load %320 : !llvm.ptr -> !llvm.ptr
  %322 = llvm.mlir.constant(1 : i32) : i32
  %323 = llvm.mlir.constant(7 : i32) : i32
  %324 = llvm.alloca %323 x !llvm.ptr : (i32) -> !llvm.ptr
  %325 = llvm.mlir.constant(0 : i32) : i32
  %326 = llvm.alloca %322 x !llvm.struct<(i1, i1, i1, vector<4xi32>)> : (i32) -> !llvm.ptr
  llvm.store %39, %326 : !llvm.struct<(i1, i1, i1, vector<4xi32>)>, !llvm.ptr
  %327 = llvm.getelementptr %324[%325] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.ptr
  llvm.store %326, %327 : !llvm.ptr, !llvm.ptr
  %328 = llvm.mlir.constant(1 : i32) : i32
  %329 = llvm.alloca %322 x !llvm.struct<(struct<(array<16 x i64>)>)> {alignment = 64 : i64} : (i32) -> !llvm.ptr
  llvm.store %96, %329 : !llvm.struct<(struct<(array<16 x i64>)>)>, !llvm.ptr
  %330 = llvm.getelementptr %324[%328] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.ptr
  llvm.store %329, %330 : !llvm.ptr, !llvm.ptr
  %331 = llvm.mlir.constant(2 : i32) : i32
  %332 = llvm.alloca %322 x !llvm.struct<(struct<()>, struct<(struct<(i32, i32, i32)>, struct<()>)>)> : (i32) -> !llvm.ptr
  llvm.store %101, %332 : !llvm.struct<(struct<()>, struct<(struct<(i32, i32, i32)>, struct<()>)>)>, !llvm.ptr
  %333 = llvm.getelementptr %324[%331] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.ptr
  llvm.store %332, %333 : !llvm.ptr, !llvm.ptr
  %334 = llvm.mlir.constant(3 : i32) : i32
  %335 = llvm.alloca %322 x !llvm.struct<(struct<(array<16 x i64>)>)> {alignment = 64 : i64} : (i32) -> !llvm.ptr
  llvm.store %158, %335 : !llvm.struct<(struct<(array<16 x i64>)>)>, !llvm.ptr
  %336 = llvm.getelementptr %324[%334] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.ptr
  llvm.store %335, %336 : !llvm.ptr, !llvm.ptr
  %337 = llvm.mlir.constant(4 : i32) : i32
  %338 = llvm.alloca %322 x !llvm.struct<(struct<()>, struct<(struct<(i32, i32, i32)>, struct<()>)>)> : (i32) -> !llvm.ptr
  llvm.store %162, %338 : !llvm.struct<(struct<()>, struct<(struct<(i32, i32, i32)>, struct<()>)>)>, !llvm.ptr
  %339 = llvm.getelementptr %324[%337] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.ptr
  llvm.store %338, %339 : !llvm.ptr, !llvm.ptr
  %340 = llvm.mlir.constant(5 : i32) : i32
  %341 = llvm.alloca %322 x !llvm.struct<(struct<(array<16 x i64>)>)> {alignment = 64 : i64} : (i32) -> !llvm.ptr
  llvm.store %219, %341 : !llvm.struct<(struct<(array<16 x i64>)>)>, !llvm.ptr
  %342 = llvm.getelementptr %324[%340] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.ptr
  llvm.store %341, %342 : !llvm.ptr, !llvm.ptr
  %343 = llvm.mlir.constant(6 : i32) : i32
  %344 = llvm.alloca %322 x !llvm.struct<(struct<()>, struct<(struct<(i32, i32, i32)>, struct<()>)>)> : (i32) -> !llvm.ptr
  llvm.store %223, %344 : !llvm.struct<(struct<()>, struct<(struct<(i32, i32, i32)>, struct<()>)>)>, !llvm.ptr
  %345 = llvm.getelementptr %324[%343] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.ptr
  llvm.store %344, %345 : !llvm.ptr, !llvm.ptr
  %346 = llvm.mlir.addressof @kernels_kernel_cutlass_kernel__upstream_blackwell_gemmDenseGemmKernel_object_at__TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom_ThrID10_ShapeMNK12812816_TVLayoutA1128161281128_TVLayoutB_0 : !llvm.ptr
  %347 = llvm.load %346 : !llvm.ptr -> !llvm.ptr
  %348 = llvm.call @_cudaLaunchKernelEx(%321, %347, %324) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  %349 = llvm.mlir.constant(0 : i32) : i32
  %350 = llvm.icmp "eq" %348, %349 : i32
  llvm.cond_br %350, ^bb25, ^bb26
^bb25:  // pred: ^bb24
  llvm.br ^bb27
^bb26:  // pred: ^bb24
  llvm.return %348 : i32
^bb27:  // pred: ^bb25
  llvm.return %34 : i32
}

llvm.func @_mlir_ciface_cutlass___call____upstream_blackwell_gemmDenseGemmKernel_object_at__Tensorgmemoi641i64_Tensorgmemoi641i64_Tensorgmemoi641i64_CUstream0xc44a5a0_functionDenseGemmKernellambdaat(%arg0: !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)>, %arg1: !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)>, %arg2: !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)>, %arg3: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface} {
  %0 = llvm.call @cutlass___call____upstream_blackwell_gemmDenseGemmKernel_object_at__Tensorgmemoi641i64_Tensorgmemoi641i64_Tensorgmemoi641i64_CUstream0xc44a5a0_functionDenseGemmKernellambdaat(%arg0, %arg1, %arg2, %arg3) : (!llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)>, !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)>, !llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i64, i64)>)>)>, !llvm.ptr) -> i32
  llvm.return %0 : i32
}
