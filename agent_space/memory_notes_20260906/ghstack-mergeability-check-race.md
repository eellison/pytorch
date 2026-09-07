---
name: ghstack-mergeability-check-race
description: "ghstack-mergeability-check can fail spuriously when it fires between two pushes; how to tell a stale race from a real desync in about ten seconds"
metadata:
  node_type: memory
  type: project
  modified: 2026-09-03
---

`Check mergeability of ghstack PR / ghstack-mergeability-check` runs
`trymerge.py --check-mergeability`, which compares each stacked PR's head-branch
diff against the corresponding `gh/USER/N/orig` commit. If a second `ghstack`
invocation (for example a follow-up `ghstack -u` to push an updated PR
description) lands while the check is in flight, the check reads the *pre-amend*
orig and fails with:

    RuntimeError: PR <n> is out of sync with the corresponding revision <sha>
    on branch gh/USER/N/orig ... This usually happens because there is a non
    ghstack change in the PR.

The message's "non ghstack change" wording is misleading in this case, and the
`<sha>` it prints will not match the branch's current head. Confirm it is stale
rather than real by hashing the two diffs -- they must be identical:

    git fetch origin 'refs/heads/gh/USER/N/*:refs/remotes/origin/gh/USER/N/*'
    git diff origin/gh/USER/N/base origin/gh/USER/N/head | git hash-object --stdin
    git diff <orig-sha>~1 <orig-sha> | git hash-object --stdin

Also cross-check the check-run's `started_at` against the orig commit date from
`gh api repos/pytorch/pytorch/commits/gh/USER/N/orig`; a race shows the push
landing seconds before the check ran. A stale failure clears on the next push
to the stack and does not require any fix to the commits.

Observed 2026-09-03 on PR #195930, and confirmed: a plain rebase onto
origin/main plus `ghstack` turned the check green on both PRs with no change
to the commits' content -- see [[to-padded-blocked-prim]].
