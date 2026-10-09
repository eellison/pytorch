# Redispatching an op that holds a keyed site: design note, before implementing

## The case
Flash-Next's QSA attention op contains a harvested call (reshape_and_cache_flash). FZ12 made the keyed site own its
nodes and the op's selector own the op's other launches. But `_redo` refuses a redo whose trace records sites
(`tr.sites` -> "dispatches otherwise"), so the op's own guards behave as graph guards and every flip of them retraces.
Repro: integration/serve/repro/core_gaps_repro.py gap_5. With FZ12 it takes 2 traces; the cause is the op's guard
x.shape[0] > 16.

## Proposal: the variant's own site serves the redo's nested call (no native change)
The fuzzfix sketch has native entries refer to a keyed site's table. That already happens without an entry: the
variant's site evaluates its key from the call's rows at every call, and `_fill` binds or learns a missing key. So if
the redo's nested call is the variant's site at the call (the same call in the same role), the entry needs only the
op's other launches, exactly the nodes the selector owns since FZ12. The site keeps serving its key, and the entry
replaces the rest.

- In `_redo`, a trace with sites is accepted when it records exactly the variant op's sites, one to one in order, each
  matching on:
  - the provider and op;
  - dtypes, ranks and scalars;
  - operands renamed onto the variant's roots (back mapping) with the same view formulas as the variant site's
    operands (renamed layouts equal, under the redo's pins as `same()` compares);
  - the same scratch buffers by index.

  Then the site's key rows are the variant's own formulas, so at every call where the entry's predicate holds, the
  site's table row is the one a trace at that call would bind. Any mismatch refuses ("another keyed site"), counted,
  and the call retraces as today.
- The entry's launches are the redo's launches minus the site's nodes, compared against the selector's nodes (the op's
  other launches) as today: a plain entry, or a section when their count or topology differs.
  - A section replaces a chain. With the site's nodes in the middle of the op's launches, the selector's nodes are two
    chains, so a section is accepted only when they are one chain (the site at the start or the end of the op's
    launches); else it refuses, counted.
- The site's own allocations (scratch) are the variant's, untouched. check_section and check_plan run on the section
  as for any section. The site's operands are roots the op holds, so liveness is unchanged.
- Redo templates: an op with sites is not templated at first. Its key would need the site's binding state, which is
  per-call table state, and verify mode would compare it. A follow-up if it matters.

## Why not the sketch's native table reference
An entry that carries its own reference to a site's table would duplicate what the site already does at every call.
It would also need native entry and arm plumbing for a case where the table, topology and scratch are the variant
site's own. The proposal changes only `_redo`'s acceptance and the launch matching; no native structure changes.

## Base and test
On top of FZ12/FZ12b (current tops step280/275, after RD2/RD3), as a separate patch.
- gap_5: 1 trace plus redispatches, bitwise, no retrace cause.
- A refusal test: the nested call's operand is another view at the flip, so it must still retrace, counted.
- check_plan clean.
