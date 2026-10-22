# vLLM decode step profiles (candidate 2, Qwen3-8B, GPU 0): default vs V (cpp closed)

torch.profiler (CPU+CUDA, with_stack) around 3 engine steps of a steady pure-decode batch; the middle step listed. Host times are inflated by the profiler's Python stack collection; use them for proportions only. Raw: out/prof_default/, out/prof_Vcpp_closed/ (chrome traces), tool: prof_list.py. async_scheduling=True (vLLM default).

## out/prof_default/decode_8.json
# out/prof_default/decode_8.json step 1: host 5624 us, GPU span 5590 us
graph launches: 1 (542 kernels inside); outside graphs: 21 (kernels 16, memcpy 4, memset 1)
| # | when | GPU us | kind | name | runtime call (corr) | host us into step | Python frames |
|---|---|---|---|---|---|---|---|
| 0 | before graph | 1.1 | kernel | void at::native::vectorized_elementwise_kernel<8, at::native::FillFunc | cudaLaunchKernel (522) | 1217 | vllm/v1/worker/gpu/model_runner.py(1159): prepare_inputs < vllm/v1/worker/gpu/model_runner.py(1516): execute_model < vllm/v1/worker/gpu_worker.py(1107): execute |
| 1 | before graph | 0.9 | HtoD (Pinned -> Device) | 64 | cudaMemcpyAsync (554) | 1385 | vllm/v1/worker/gpu/buffer_utils.py(26): async_copy_to_gpu < vllm/v1/worker/gpu/model_runner.py(1159): prepare_inputs < vllm/v1/worker/gpu/model_runner.py(1516): |
| 2 | before graph | 1.1 | kernel | void (anonymous namespace)::elementwise_kernel_with_index<int, (anonym | cudaLaunchKernel (579) | 1483 | vllm/v1/worker/gpu/model_runner.py(1159): prepare_inputs < vllm/v1/worker/gpu/model_runner.py(1516): execute_model < vllm/v1/worker/gpu_worker.py(1107): execute |
| 3 | before graph | 0.9 | memset | 32 | cudaMemsetAsync (591) | 1531 | vllm/v1/worker/gpu/model_runner.py(1159): prepare_inputs < vllm/v1/worker/gpu/model_runner.py(1516): execute_model < vllm/v1/worker/gpu_worker.py(1107): execute |
| 4 | before graph | 1.1 | HtoD (Pinned -> Device) | 4100 | cudaMemcpyAsync (608) | 1658 | vllm/v1/worker/gpu/buffer_utils.py(26): async_copy_to_gpu < vllm/v1/worker/gpu/model_runner.py(1159): prepare_inputs < vllm/v1/worker/gpu/model_runner.py(1516): |
| 5 | before graph | 2.0 | kernel | _prepare_pos_seq_lens_kernel | cuLaunchKernelEx (629) | 1796 | vllm/v1/worker/gpu/input_batch.py(367): prepare_pos_seq_lens < vllm/v1/worker/gpu/model_runner.py(1159): prepare_inputs < vllm/v1/worker/gpu/model_runner.py(151 |
| 6 | before graph | 2.6 | kernel | _combine_sampled_and_draft_tokens_kernel | cuLaunchKernelEx (650) | 1966 | vllm/v1/worker/gpu/input_batch.py(449): combine_sampled_and_draft_tokens < vllm/v1/worker/gpu/model_runner.py(1159): prepare_inputs < vllm/v1/worker/gpu/model_r |
| 7 | before graph | 3.6 | kernel | _gather_block_tables_kernel | cuLaunchKernelEx (667) | 2159 | vllm/v1/worker/gpu/block_table.py(149): gather_block_tables < vllm/v1/worker/gpu/model_runner.py(1368): prepare_attn < vllm/v1/worker/gpu/model_runner.py(1516): |
| 8 | before graph | 3.1 | kernel | _compute_slot_mappings_kernel | cuLaunchKernelEx (682) | 2285 | vllm/v1/worker/gpu/block_table.py(191): compute_slot_mappings < vllm/v1/worker/gpu/model_runner.py(1368): prepare_attn < vllm/v1/worker/gpu/model_runner.py(1516 |
| 9 | after graph | 2.1 | kernel | void at::native::vectorized_gather_kernel<16, long>(char*, char*, long | cudaLaunchKernel (719) | 3363 | vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens < vllm/v1/worker/gpu/eplb_utils.py(25): wrapper |
| 10 | after graph | 203.8 | kernel | nvjet_sm103_tst_512x8_64x3_2x1_v_bz_TNT | cuLaunchKernelEx (738) | 3560 | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm < vllm/model_executor/layers/vocab_parallel_embedding.py(68): apply < vllm/model_executor/laye |
| 11 | after graph | 3.1 | kernel | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_inde | cudaLaunchKernel (752) | 3664 | vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens |
| 12 | after graph | 3.0 | kernel | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_inde | cudaLaunchKernel (766) | 3712 | vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens |
| 13 | after graph | 4.2 | kernel | _gumbel_sample_kernel | cuLaunchKernelEx (795) | 4036 | vllm/v1/worker/gpu/sample/gumbel.py(251): gumbel_sample < vllm/v1/worker/gpu/sample/sampler.py(274): sample < vllm/v1/worker/gpu/sample/sampler.py(122): __call_ |
| 14 | after graph | 4.5 | kernel | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at: | cudaLaunchKernel (806) | 4102 | vllm/v1/worker/gpu/sample/gumbel.py(251): gumbel_sample < vllm/v1/worker/gpu/sample/sampler.py(274): sample < vllm/v1/worker/gpu/sample/sampler.py(122): __call_ |
| 15 | after graph | 2.6 | kernel | void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native | cudaLaunchKernel (816) | 4162 | vllm/v1/worker/gpu/sample/gumbel.py(251): gumbel_sample < vllm/v1/worker/gpu/sample/sampler.py(274): sample < vllm/v1/worker/gpu/sample/sampler.py(122): __call_ |
| 16 | after graph | 1.4 | kernel | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc | cudaLaunchKernel (830) | 4227 | vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens |
| 17 | after graph | 2.5 | kernel | _get_num_sampled_and_rejected_kernel | cuLaunchKernelEx (849) | 4344 | vllm/v1/worker/gpu/input_batch.py(523): get_num_sampled_and_rejected < vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py( |
| 18 | after graph | 3.9 | kernel | _post_update_kernel | cuLaunchKernelEx (913) | 4790 | vllm/v1/worker/gpu/input_batch.py(604): post_update < vllm/v1/worker/gpu/model_runner.py(1476): postprocess_sampled < vllm/v1/worker/gpu/model_runner.py(1816):  |
| 19 | after graph | 1.3 | DtoH (Device -> Pinned) | 64 | cudaMemcpyAsync (878) | 4552 | vllm/v1/worker/gpu/async_utils.py(253): async_copy_to_np < vllm/v1/worker/gpu/async_utils.py(116): __init__ < vllm/v1/worker/gpu/model_runner.py(1816): sample_t |
| 20 | after graph | 1.6 | DtoH (Device -> Pinned) | 32 | cudaMemcpyAsync (887) | 4630 | vllm/v1/worker/gpu/async_utils.py(253): async_copy_to_np < vllm/v1/worker/gpu/async_utils.py(116): __init__ < vllm/v1/worker/gpu/model_runner.py(1816): sample_t |

host phases (inclusive us within the step; count):
- execute_model: 2588 us (1)
- prepare_inputs: 912 us (2)
- prepare_attn: 456 us (2)
- prepare_pos_seq_lens: 108 us (1)
- combine_sampled_and_draft_tokens: 140 us (1)
- build_slot_mappings_by_layer: 35 us (1)
- async_copy_to_gpu: 210 us (2)
- apply_staged_writes: 345 us (2)
- update_requests: 21 us (1)
- add_requests: 324 us (1)
- gather_batch_req_state: 57 us (1)
- sample_tokens: 1632 us (1)
- sample: 1093 us (1)
- compute_logits: 222 us (1)
- run_fullgraph: 552 us (1)
- forward: 197 us (1)
- schedule: 605 us (1)
- update_from_output: 326 us (9)
- process_outputs: 175 us (1)
- step_with_batch_queue: 5288 us (1)

## out/prof_Vcpp_closed/decode_8.json
# out/prof_Vcpp_closed/decode_8.json step 1: host 4977 us, GPU span 5111 us
graph launches: 1 (548 kernels inside); outside graphs: 13 (kernels 10, memcpy 3, memset 0)
| # | when | GPU us | kind | name | runtime call (corr) | host us into step | Python frames |
|---|---|---|---|---|---|---|---|
| 0 | before graph | 2.1 | HtoD (Pinned -> Device) | 28688 | cudaMemcpyAsync (296) | 1083 | hostcuts.py(64): prepare_inputs < adapter.py(317): prepare_inputs < vllm/v1/worker/gpu/model_runner.py(1516): execute_model |
| 1 | after graph | 2.2 | kernel | void at::native::vectorized_gather_kernel<16, long>(char*, char*, long | cudaLaunchKernel (356) | 2381 | vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens < vllm/v1/worker/gpu/eplb_utils.py(25): wrapper |
| 2 | after graph | 200.3 | kernel | nvjet_sm103_tst_512x8_64x3_2x1_v_bz_TNT | cuLaunchKernelEx (375) | 2603 | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm < vllm/model_executor/layers/vocab_parallel_embedding.py(68): apply < vllm/model_executor/laye |
| 3 | after graph | 3.3 | kernel | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_inde | cudaLaunchKernel (389) | 2725 | vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens |
| 4 | after graph | 3.0 | kernel | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_inde | cudaLaunchKernel (403) | 2785 | vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens |
| 5 | after graph | 5.2 | kernel | _gumbel_sample_kernel | cuLaunchKernelEx (432) | 3185 | vllm/v1/worker/gpu/sample/gumbel.py(251): gumbel_sample < vllm/v1/worker/gpu/sample/sampler.py(274): sample < vllm/v1/worker/gpu/sample/sampler.py(122): __call_ |
| 6 | after graph | 4.8 | kernel | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at: | cudaLaunchKernel (443) | 3259 | vllm/v1/worker/gpu/sample/gumbel.py(251): gumbel_sample < vllm/v1/worker/gpu/sample/sampler.py(274): sample < vllm/v1/worker/gpu/sample/sampler.py(122): __call_ |
| 7 | after graph | 2.1 | kernel | void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native | cudaLaunchKernel (453) | 3330 | vllm/v1/worker/gpu/sample/gumbel.py(251): gumbel_sample < vllm/v1/worker/gpu/sample/sampler.py(274): sample < vllm/v1/worker/gpu/sample/sampler.py(122): __call_ |
| 8 | after graph | 1.6 | kernel | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc | cudaLaunchKernel (467) | 3412 | vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens |
| 9 | after graph | 2.8 | kernel | _get_num_sampled_and_rejected_kernel | cuLaunchKernelEx (486) | 3565 | vllm/v1/worker/gpu/input_batch.py(523): get_num_sampled_and_rejected < vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py( |
| 10 | after graph | 4.7 | kernel | _post_update_kernel | cuLaunchKernelEx (551) | 4088 | vllm/v1/worker/gpu/input_batch.py(604): post_update < vllm/v1/worker/gpu/model_runner.py(1476): postprocess_sampled < vllm/v1/worker/gpu/model_runner.py(1816):  |
| 11 | after graph | 2.0 | DtoH (Device -> Pinned) | 64 | cudaMemcpyAsync (516) | 3805 | vllm/v1/worker/gpu/async_utils.py(253): async_copy_to_np < vllm/v1/worker/gpu/async_utils.py(116): __init__ < vllm/v1/worker/gpu/model_runner.py(1816): sample_t |
| 12 | after graph | 1.3 | DtoH (Device -> Pinned) | 32 | cudaMemcpyAsync (525) | 3888 | vllm/v1/worker/gpu/async_utils.py(253): async_copy_to_np < vllm/v1/worker/gpu/async_utils.py(116): __init__ < vllm/v1/worker/gpu/model_runner.py(1816): sample_t |

host phases (inclusive us within the step; count):
- execute_model: 1540 us (1)
- prepare_inputs: 308 us (2)
- prepare_attn: 5 us (1)
- build_slot_mappings_by_layer: 1 us (1)
- apply_staged_writes: 70 us (2)
- update_requests: 22 us (1)
- add_requests: 5 us (1)
- gather_batch_req_state: 76 us (1)
- sample_tokens: 1942 us (1)
- sample: 1325 us (1)
- compute_logits: 246 us (1)
- forward: 870 us (2)
- schedule: 642 us (1)
- update_from_output: 327 us (9)
- process_outputs: 195 us (1)
- step_with_batch_queue: 4595 us (1)

## out/prof_default/decode_64.json
# out/prof_default/decode_64.json step 1: host 10193 us, GPU span 6308 us
graph launches: 1 (470 kernels inside); outside graphs: 21 (kernels 16, memcpy 4, memset 1)
| # | when | GPU us | kind | name | runtime call (corr) | host us into step | Python frames |
|---|---|---|---|---|---|---|---|
| 0 | before graph | 1.6 | kernel | void at::native::vectorized_elementwise_kernel<8, at::native::FillFunc | cudaLaunchKernel (23804) | 3526 | vllm/v1/worker/gpu/model_runner.py(1159): prepare_inputs < vllm/v1/worker/gpu/model_runner.py(1516): execute_model < vllm/v1/worker/gpu_worker.py(1107): execute |
| 1 | before graph | 0.9 | HtoD (Pinned -> Device) | 512 | cudaMemcpyAsync (23836) | 3747 | vllm/v1/worker/gpu/buffer_utils.py(26): async_copy_to_gpu < vllm/v1/worker/gpu/model_runner.py(1159): prepare_inputs < vllm/v1/worker/gpu/model_runner.py(1516): |
| 2 | before graph | 1.5 | kernel | void (anonymous namespace)::elementwise_kernel_with_index<int, (anonym | cudaLaunchKernel (23861) | 3857 | vllm/v1/worker/gpu/model_runner.py(1159): prepare_inputs < vllm/v1/worker/gpu/model_runner.py(1516): execute_model < vllm/v1/worker/gpu_worker.py(1107): execute |
| 3 | before graph | 0.9 | memset | 256 | cudaMemsetAsync (23873) | 3910 | vllm/v1/worker/gpu/model_runner.py(1159): prepare_inputs < vllm/v1/worker/gpu/model_runner.py(1516): execute_model < vllm/v1/worker/gpu_worker.py(1107): execute |
| 4 | before graph | 1.1 | HtoD (Pinned -> Device) | 4100 | cudaMemcpyAsync (23890) | 4041 | vllm/v1/worker/gpu/buffer_utils.py(26): async_copy_to_gpu < vllm/v1/worker/gpu/model_runner.py(1159): prepare_inputs < vllm/v1/worker/gpu/model_runner.py(1516): |
| 5 | before graph | 1.9 | kernel | _prepare_pos_seq_lens_kernel | cuLaunchKernelEx (23911) | 4197 | vllm/v1/worker/gpu/input_batch.py(367): prepare_pos_seq_lens < vllm/v1/worker/gpu/model_runner.py(1159): prepare_inputs < vllm/v1/worker/gpu/model_runner.py(151 |
| 6 | before graph | 2.8 | kernel | _combine_sampled_and_draft_tokens_kernel | cuLaunchKernelEx (23932) | 4356 | vllm/v1/worker/gpu/input_batch.py(449): combine_sampled_and_draft_tokens < vllm/v1/worker/gpu/model_runner.py(1159): prepare_inputs < vllm/v1/worker/gpu/model_r |
| 7 | before graph | 3.5 | kernel | _gather_block_tables_kernel | cuLaunchKernelEx (23949) | 4531 | vllm/v1/worker/gpu/block_table.py(149): gather_block_tables < vllm/v1/worker/gpu/model_runner.py(1368): prepare_attn < vllm/v1/worker/gpu/model_runner.py(1516): |
| 8 | before graph | 3.5 | kernel | _compute_slot_mappings_kernel | cuLaunchKernelEx (23964) | 4654 | vllm/v1/worker/gpu/block_table.py(191): compute_slot_mappings < vllm/v1/worker/gpu/model_runner.py(1368): prepare_attn < vllm/v1/worker/gpu/model_runner.py(1516 |
| 9 | after graph | 2.3 | kernel | void at::native::vectorized_gather_kernel<16, long>(char*, char*, long | cudaLaunchKernel (24001) | 5707 | vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens < vllm/v1/worker/gpu/eplb_utils.py(25): wrapper |
| 10 | after graph | 208.5 | kernel | nvjet_sm103_tst_512x64_64x3_2x1_2cta_v_bz_TNT | cuLaunchKernelEx (24020) | 5906 | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm < vllm/model_executor/layers/vocab_parallel_embedding.py(68): apply < vllm/model_executor/laye |
| 11 | after graph | 3.2 | kernel | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_inde | cudaLaunchKernel (24034) | 5998 | vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens |
| 12 | after graph | 3.1 | kernel | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_inde | cudaLaunchKernel (24048) | 6041 | vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens |
| 13 | after graph | 13.5 | kernel | _gumbel_sample_kernel | cuLaunchKernelEx (24077) | 6354 | vllm/v1/worker/gpu/sample/gumbel.py(251): gumbel_sample < vllm/v1/worker/gpu/sample/sampler.py(274): sample < vllm/v1/worker/gpu/sample/sampler.py(122): __call_ |
| 14 | after graph | 4.3 | kernel | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at: | cudaLaunchKernel (24088) | 6412 | vllm/v1/worker/gpu/sample/gumbel.py(251): gumbel_sample < vllm/v1/worker/gpu/sample/sampler.py(274): sample < vllm/v1/worker/gpu/sample/sampler.py(122): __call_ |
| 15 | after graph | 2.3 | kernel | void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native | cudaLaunchKernel (24098) | 6467 | vllm/v1/worker/gpu/sample/gumbel.py(251): gumbel_sample < vllm/v1/worker/gpu/sample/sampler.py(274): sample < vllm/v1/worker/gpu/sample/sampler.py(122): __call_ |
| 16 | after graph | 1.1 | kernel | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc | cudaLaunchKernel (24112) | 6531 | vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens |
| 17 | after graph | 2.5 | kernel | _get_num_sampled_and_rejected_kernel | cuLaunchKernelEx (24131) | 6643 | vllm/v1/worker/gpu/input_batch.py(523): get_num_sampled_and_rejected < vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py( |
| 18 | after graph | 3.4 | kernel | _post_update_kernel | cuLaunchKernelEx (24195) | 7084 | vllm/v1/worker/gpu/input_batch.py(604): post_update < vllm/v1/worker/gpu/model_runner.py(1476): postprocess_sampled < vllm/v1/worker/gpu/model_runner.py(1816):  |
| 19 | after graph | 1.4 | DtoH (Device -> Pinned) | 512 | cudaMemcpyAsync (24160) | 6855 | vllm/v1/worker/gpu/async_utils.py(253): async_copy_to_np < vllm/v1/worker/gpu/async_utils.py(116): __init__ < vllm/v1/worker/gpu/model_runner.py(1816): sample_t |
| 20 | after graph | 1.3 | DtoH (Device -> Pinned) | 256 | cudaMemcpyAsync (24169) | 6930 | vllm/v1/worker/gpu/async_utils.py(253): async_copy_to_np < vllm/v1/worker/gpu/async_utils.py(116): __init__ < vllm/v1/worker/gpu/model_runner.py(1816): sample_t |

host phases (inclusive us within the step; count):
- execute_model: 2731 us (1)
- prepare_inputs: 986 us (2)
- prepare_attn: 456 us (2)
- prepare_pos_seq_lens: 129 us (1)
- combine_sampled_and_draft_tokens: 136 us (1)
- build_slot_mappings_by_layer: 43 us (1)
- async_copy_to_gpu: 247 us (2)
- apply_staged_writes: 366 us (2)
- update_requests: 35 us (1)
- add_requests: 346 us (1)
- gather_batch_req_state: 110 us (1)
- sample_tokens: 1574 us (1)
- sample: 1042 us (1)
- compute_logits: 220 us (1)
- run_fullgraph: 514 us (1)
- forward: 195 us (1)
- schedule: 2788 us (1)
- update_from_output: 1814 us (65)
- process_outputs: 949 us (1)
- step_with_batch_queue: 9040 us (1)

## out/prof_Vcpp_closed/decode_64.json
# out/prof_Vcpp_closed/decode_64.json step 1: host 10177 us, GPU span 5731 us
graph launches: 1 (476 kernels inside); outside graphs: 13 (kernels 10, memcpy 3, memset 0)
| # | when | GPU us | kind | name | runtime call (corr) | host us into step | Python frames |
|---|---|---|---|---|---|---|---|
| 0 | before graph | 2.0 | HtoD (Pinned -> Device) | 28688 | cudaMemcpyAsync (73031) | 3452 | hostcuts.py(64): prepare_inputs < adapter.py(317): prepare_inputs < vllm/v1/worker/gpu/model_runner.py(1516): execute_model |
| 1 | after graph | 2.2 | kernel | void at::native::vectorized_gather_kernel<16, long>(char*, char*, long | cudaLaunchKernel (73091) | 4872 | vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens < vllm/v1/worker/gpu/eplb_utils.py(25): wrapper |
| 2 | after graph | 206.6 | kernel | nvjet_sm103_tst_512x64_64x3_2x1_2cta_v_bz_TNT | cuLaunchKernelEx (73110) | 5136 | vllm/model_executor/layers/utils.py(92): default_unquantized_gemm < vllm/model_executor/layers/vocab_parallel_embedding.py(68): apply < vllm/model_executor/laye |
| 3 | after graph | 3.5 | kernel | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_inde | cudaLaunchKernel (73124) | 5264 | vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens |
| 4 | after graph | 3.0 | kernel | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_inde | cudaLaunchKernel (73138) | 5321 | vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens |
| 5 | after graph | 15.7 | kernel | _gumbel_sample_kernel | cuLaunchKernelEx (73167) | 5709 | vllm/v1/worker/gpu/sample/gumbel.py(251): gumbel_sample < vllm/v1/worker/gpu/sample/sampler.py(274): sample < vllm/v1/worker/gpu/sample/sampler.py(122): __call_ |
| 6 | after graph | 5.1 | kernel | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at: | cudaLaunchKernel (73178) | 5786 | vllm/v1/worker/gpu/sample/gumbel.py(251): gumbel_sample < vllm/v1/worker/gpu/sample/sampler.py(274): sample < vllm/v1/worker/gpu/sample/sampler.py(122): __call_ |
| 7 | after graph | 2.3 | kernel | void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native | cudaLaunchKernel (73188) | 5856 | vllm/v1/worker/gpu/sample/gumbel.py(251): gumbel_sample < vllm/v1/worker/gpu/sample/sampler.py(274): sample < vllm/v1/worker/gpu/sample/sampler.py(122): __call_ |
| 8 | after graph | 1.6 | kernel | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc | cudaLaunchKernel (73202) | 5934 | vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py(1398): sample < vllm/v1/worker/gpu/model_runner.py(1816): sample_tokens |
| 9 | after graph | 3.0 | kernel | _get_num_sampled_and_rejected_kernel | cuLaunchKernelEx (73221) | 6077 | vllm/v1/worker/gpu/input_batch.py(523): get_num_sampled_and_rejected < vllm/v1/worker/gpu/sample/sampler.py(122): __call__ < vllm/v1/worker/gpu/model_runner.py( |
| 10 | after graph | 4.0 | kernel | _post_update_kernel | cuLaunchKernelEx (73286) | 6674 | vllm/v1/worker/gpu/input_batch.py(604): post_update < vllm/v1/worker/gpu/model_runner.py(1476): postprocess_sampled < vllm/v1/worker/gpu/model_runner.py(1816):  |
| 11 | after graph | 1.3 | DtoH (Device -> Pinned) | 512 | cudaMemcpyAsync (73251) | 6408 | vllm/v1/worker/gpu/async_utils.py(253): async_copy_to_np < vllm/v1/worker/gpu/async_utils.py(116): __init__ < vllm/v1/worker/gpu/model_runner.py(1816): sample_t |
| 12 | after graph | 1.3 | DtoH (Device -> Pinned) | 256 | cudaMemcpyAsync (73260) | 6496 | vllm/v1/worker/gpu/async_utils.py(253): async_copy_to_np < vllm/v1/worker/gpu/async_utils.py(116): __init__ < vllm/v1/worker/gpu/model_runner.py(1816): sample_t |

host phases (inclusive us within the step; count):
- execute_model: 1778 us (1)
- prepare_inputs: 348 us (2)
- prepare_attn: 5 us (1)
- build_slot_mappings_by_layer: 1 us (1)
- apply_staged_writes: 108 us (2)
- update_requests: 45 us (1)
- add_requests: 5 us (1)
- gather_batch_req_state: 110 us (1)
- sample_tokens: 2065 us (1)
- sample: 1392 us (1)
- compute_logits: 277 us (1)
- forward: 963 us (2)
- schedule: 2862 us (1)
- update_from_output: 1989 us (65)
- process_outputs: 1083 us (1)
- step_with_batch_queue: 8850 us (1)
