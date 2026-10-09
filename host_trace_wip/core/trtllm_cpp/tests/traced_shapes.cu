// The traced build's TMA shape and stride lists (KernelParams::makeTmaShapeStride*, templates over the traced
// runner params) have their element counts: the cudafe brace-list bug (test_nvcc_vector_init.cu) built them with
// one element. Prints "Q a b O c d K e f", which tests/test_drift.py checks. CPU only.
#include "../ht/ht_traced.cu"

int main() {
  fi_ht_traced::TllmGenFmhaRunnerParams p;
  p.mQkvLayout = QkvLayout::PagedKv;
  p.mHeadDimQk = 128;
  p.mHeadDimV = 128;
  p.mNumHeadsQ = 32;
  p.mNumHeadsKv = 8;
  p.mNumHeadsQPerKv = 4;
  p.mSumOfSeqLensQ = 4;
  p.mBatchSize = 4;
  p.mMaxSeqLenKv = 300;
  p.mNumTokensPerPage = 16;
  p.mNumPagesInMemPool = 4096;
  p.kStrideKeysValues = 128;
  p.kStrideHeads = 2048;
  p.kStrideBatch = 16384;
  auto [sq, tq, bq, nq] = fi_ht_traced::KernelParams::makeTmaShapeStrideQ(p, true, true, 8, 8, 64);
  auto [so, to] = fi_ht_traced::KernelParams::makeTmaShapeStrideO(p);
  fi_ht_traced::KernelParams kp;
  auto [sk, tk] = fi_ht_traced::KernelParams::makeTmaShapeStrideKv(p, kp, DATA_TYPE_BF16, true, false, 1);
  printf("Q %zu %zu O %zu %zu K %zu %zu\n", sq.size(), tq.size(), so.size(), to.size(), sk.size(), tk.size());
  return 0;
}
