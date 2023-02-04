
import torch
import dataclasses
from typing import List, Any, Dict, Literal, Tuple, Optional

def cudagraphify(model, static_inputs):
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # copy static_inputs because it will be cleared in model
    with torch.cuda.stream(stream):
        model(list(static_inputs))
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    torch.cuda.memory._record_memory_history(True,
        # keep 100,000 alloc/free events from before the snapshot
        trace_alloc_max_entries=100000,

        # record stack information for the trace events
        trace_alloc_record_context=True)

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = model(list(static_inputs))
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    return graph, static_outputs


def model(args):
    x, y = args
    args.clear()
    z = x + y
    for _ in range(2):
        z = z + 10

    return z    

def cudagraph_no_warmup(fn):
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    torch.cuda.memory._record_memory_history(True,
        # keep 100,000 alloc/free events from before the snapshot
        trace_alloc_max_entries=1,

        # record stack information for the trace events
        trace_alloc_record_context=False)
    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = fn()
    return graph, static_outputs


inps = [torch.rand([40, 40], device="cuda")]
inps = inps + inps

def model(args):
    x, y = args
    z = x + y 
    z = z + x
    
def fn():
    print("RUNNING X")
    x = torch.zeros(int(256e6 // 4), dtype=torch.int, device='cuda')
    return x

graph, outputs = cudagraph_no_warmup(fn)
print("DEL OUTPUTS")
del outputs

# # cudagraphify(model, inps)

snapshot = torch.cuda.memory._snapshot()


assert all(trace == [] for trace in snapshot["device_traces"][1:])

@dataclasses.dataclass
class TraceEvent(object):
    frames: List[Dict[str, Any]]
    action: str
    addr: int
    size: int
    stream: int

@dataclasses.dataclass
class Block(object):
    size: int
    state: Any
    history: Any



@dataclasses.dataclass
class Segment(object):
    device: int
    address: int
    total_size: int
    allocated_size: int
    active_size: int
    stream: int
    segment_type: Literal["large", "small"]
    blocks: List[Block]
    segment_pool_id: Optional[Tuple[int, int]]


blocks = snapshot["segments"][0]["blocks"]
print(blocks[0]["state"])

breakpoint()

del graph
del outputs

torch.rand([4], device="cuda")


for event in snapshot["device_traces"][0]:
    event["frames"] = []
    t_event = TraceEvent(**event)


# The blocks must already present in the state already 
# so, we just need a way to split them


# Absorb 





