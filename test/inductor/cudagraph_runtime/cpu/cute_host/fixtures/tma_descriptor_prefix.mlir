module {
llvm.func @cutlass___call____upstream_tma_v0Sm100SimpleCopyKernel_object_at__Tensorgmemodiv16i64div161_Tensorgmemodiv16i64div161(%arg0: !llvm.struct<(ptr<1>, struct<(struct<(i32, i32)>, i64)>)>, %arg1: !llvm.struct<(ptr<1>, struct<(struct<(i32, i32)>, i64)>)>) -> !llvm.struct<(struct<(array<16 x i64>)>)> attributes {llvm.emit_c_interface} {
  %0 = llvm.mlir.constant(2 : i32) : i32
  %1 = llvm.mlir.constant(4 : i32) : i32
  %2 = llvm.mlir.constant(17 : i32) : i32
  %5 = llvm.mlir.constant(6 : i32) : i32
  %6 = llvm.mlir.constant(false) : i1
  %7 = llvm.mlir.poison : !llvm.struct<(struct<()>, struct<(struct<(i32, i32)>, struct<()>)>)>
  %8 = llvm.mlir.poison : !llvm.struct<(struct<(i32, i32)>, struct<()>)>
  %9 = llvm.mlir.poison : !llvm.struct<(struct<(array<16 x i64>)>)>
  %10 = llvm.mlir.constant(127 : i64) : i64
  %11 = llvm.mlir.constant(9151314442816847872 : i64) : i64
  %12 = llvm.mlir.constant(262930 : i64) : i64
  %13 = llvm.mlir.constant(0 : i32) : i32
  %14 = llvm.mlir.constant(1 : i32) : i32
  %15 = llvm.mlir.constant(32 : i32) : i32
  %16 = llvm.mlir.constant(33792 : i64) : i64
  %17 = llvm.mlir.constant(128 : i32) : i32
  %18 = llvm.mlir.poison : !llvm.struct<()>
  %19 = llvm.mlir.constant(15 : i64) : i64
  %20 = llvm.mlir.constant(36 : i64) : i64
  %21 = llvm.mlir.constant(32 : i64) : i64
  %22 = llvm.mlir.constant(21 : i64) : i64
  %23 = llvm.mlir.constant(131072 : i64) : i64
  %24 = llvm.mlir.constant(8 : i64) : i64
  %25 = llvm.mlir.constant(9007199254740991 : i64) : i64
  %26 = llvm.mlir.constant(16 : i64) : i64
  %27 = llvm.mlir.constant(4 : i64) : i64
  %28 = llvm.mlir.constant(4294967295 : i64) : i64
  %29 = llvm.mlir.constant(2 : i64) : i64
  %30 = llvm.mlir.constant(1 : i64) : i64
  %31 = llvm.mlir.constant(0 : i64) : i64
  %32 = llvm.mlir.constant(16 : i32) : i32
  %33 = llvm.alloca %32 x i64 {alignment = 64 : i64} : (i32) -> !llvm.ptr
  %34 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32)>, i64)>)> 
  %35 = llvm.extractvalue %arg0[1, 0, 0] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32)>, i64)>)> 
  %36 = llvm.extractvalue %arg0[1, 0, 1] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32)>, i64)>)> 
  %37 = llvm.extractvalue %arg0[1, 1] : !llvm.struct<(ptr<1>, struct<(struct<(i32, i32)>, i64)>)> 
  %38 = llvm.sext %36 : i32 to i64
  %39 = llvm.sext %35 : i32 to i64
  %40 = llvm.mul %37, %29 : i64
  %41 = llvm.ptrtoint %34 : !llvm.ptr<1> to i64
  %42 = llvm.getelementptr %33[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %31, %42 : i64, !llvm.ptr
  %43 = llvm.getelementptr %33[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %31, %43 : i64, !llvm.ptr
  %44 = llvm.getelementptr %33[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %31, %44 : i64, !llvm.ptr
  %45 = llvm.getelementptr %33[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %31, %45 : i64, !llvm.ptr
  %46 = llvm.getelementptr %33[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %31, %46 : i64, !llvm.ptr
  %47 = llvm.getelementptr %33[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %31, %47 : i64, !llvm.ptr
  %48 = llvm.getelementptr %33[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %31, %48 : i64, !llvm.ptr
  %49 = llvm.getelementptr %33[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i64>
  llvm.store %31, %49 : i64, !llvm.ptr
  %50 = llvm.udiv %41, %26 : i64
  %51 = llvm.and %50, %25 : i64
  %52 = llvm.shl %51, %27 : i64
  llvm.store %52, %42 : i64, !llvm.ptr
  %53 = llvm.sub %39, %30 : i64
  %54 = llvm.mul %53, %40 : i64
  %55 = llvm.mul %38, %26 : i64
  %56 = llvm.udiv %55, %24 : i64
  %57 = llvm.add %56, %54 : i64
  %58 = llvm.icmp "uge" %57, %23 : i64
  %59 = llvm.zext %58 : i1 to i64
  %60 = llvm.shl %59, %22 : i64
  %61 = llvm.udiv %40, %26 : i64
  %62 = llvm.shl %61, %21 : i64
  %63 = llvm.or %60, %62 : i64
  %64 = llvm.or %63, %12 : i64
  llvm.store %64, %43 : i64, !llvm.ptr
  llvm.store %31, %44 : i64, !llvm.ptr
  %65 = llvm.lshr %40, %20 : i64
  %66 = llvm.and %65, %19 : i64
  %67 = llvm.shl %66, %21 : i64
  llvm.store %67, %45 : i64, !llvm.ptr
  %68 = llvm.sub %38, %30 : i64
  %69 = llvm.and %68, %28 : i64
  %70 = llvm.shl %53, %21 : i64
  %71 = llvm.or %69, %70 : i64
  llvm.store %71, %46 : i64, !llvm.ptr
  llvm.store %31, %47 : i64, !llvm.ptr
  llvm.store %11, %48 : i64, !llvm.ptr
  llvm.store %10, %49 : i64, !llvm.ptr
  %72 = llvm.ptrtoint %33 : !llvm.ptr to i64
  %73 = llvm.inttoptr %72 : i64 to !llvm.ptr
  %74 = llvm.load %73 {nontemporal} : !llvm.ptr -> !llvm.struct<(array<16 x i64>)>
  %75 = llvm.insertvalue %74, %9[0] : !llvm.struct<(struct<(array<16 x i64>)>)> 
  llvm.return %75 : !llvm.struct<(struct<(array<16 x i64>)>)>
}
}
