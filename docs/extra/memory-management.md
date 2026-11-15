# Memory Management in DJL-Based RL

> A practical guide to avoiding memory leaks in `java-rl-dqn-to-rainbow`.

## Why this guide exists

DJL (Deep Java Library) tracks every `NDArray` in a hierarchical tree of
`NDManager` instances. An array is only truly freed when its owning
manager closes. If a manager never closes, or if a temporary array is
attached to a long-lived manager, the array leaks.

In a reinforcement learning training loop this is catastrophic: a single
leaked array per step accumulates to **millions of arrays** over a long
run, exhausting both JVM heap and native PyTorch memory. The first
indicator is usually a slow, steady climb in off-heap memory and a
multi-second GC pause every few thousand steps.

This document captures the patterns we use to keep leaks at zero, the
helpers that make those patterns ergonomic, and the test that catches
regressions automatically.

## The manager hierarchy

DJL builds a tree of NDManagers rooted at the system manager:

```
systemManager                 (always exists, never closed)
├── modelManager              (lives as long as the Model)
├── envManager                (lives as long as the Env)
└── trainingManager           (lives as long as the training session)
    └── replayBufferManager
        └── sampleManager     (created per sample(), closed by VecExperience)
```

Two simple rules follow from the tree:

- **Attach long-lived data to a long-lived manager.**
  Replay-buffer states live in the buffer's sub-manager; that is correct
  because the buffer outlives any single batch.
- **Attach short-lived data to a short-lived manager.**
  Forward outputs, intermediate activations, and sampled batches must live
  in a manager that will be closed at the end of the operation. Otherwise
  they accumulate forever in the model or training manager.

## Helpers (`DJLMemoryManagement`)

All helpers are static, side-effect-free, and intended to be used at
construction and call sites.

| Helper | Purpose | Example |
|--------|---------|---------|
| `subMgr(parent, name)` | Create a named sub-manager of `parent`. | `var sub = subMgr(parent, "noisy-layer");` |
| `subMgr(ndArray, name)` | Same, derived from the array's manager. | `var sub = subMgr(arr, "scoped-input");` |
| `subMgr(parent, Class)` | Use the simple class name as the sub-manager name. | `subMgr(parent, getClass());` |
| `setName(arr, name)` | Tag an `NDArray` with a debug-friendly name. | `setName(state, "buffer-state");` |
| `newModel(Class)` / `newModel(name, device)` | Create a `Model` whose underlying manager is named. | `newModel(getClass(), device);` |
| `debugDump(manager)` | Print the manager tree (names + resource counts). | `debugDump(parent);` |
| `managedArrayCount(manager)` | Count NDArrays in one manager (no recursion). | `managedArrayCount(parent);` |
| `systemResourceCount()` | Count **all** resources across the whole hierarchy. | `int n = systemResourceCount();` |
| `scoped(...)` (family) | Allocating pattern with a temporary sub-manager. | `scoped(it -> { ... }, input);` |

The naming convention `name + "-" + uid` means a sub created via
`subMgr(parent, "noisy-layer")` appears as
`noisy-layer-uid-139933978505100-3` in the debug dump, which is enough to
locate it in code.

## The five rules

### 1. Never attach temporaries to a long-lived manager

The single most common leak. The fix is always either: close the array
explicitly, or move it to a manager that will be closed.

```java
// BAD — `output` is now permanently in the model manager
var output = block.forward(parameterStore, inputs, training);
output.attach(modelManager);
return output;
```

### 2. Use `tempAttach` for parameters, never `attach`

When you need to operate on long-lived parameters inside a temporary
scope, attach them temporarily. When the scope closes, the parameters
are restored to their original manager with no side effects.

```java
// GOOD
@Cleanup var sub = subMgr(inputs.getManager(), "noisy-layer");
sub.tempAttachAll(wMu, wSigma, bMu, bSigma);
// intermediates are born in `sub` and cleaned up when it closes
// parameters are restored to their original manager at the end
```

### 3. Use `@Cleanup`, not manual `close()`

Lombok's `@Cleanup` guarantees `close()` is called even when an
exception is thrown. Manual close is easy to forget, and the JVM does
not warn you about it.

```java
// GOOD
@Cleanup var sub = subMgr(parent, "scoped");

// BAD
var sub = subMgr(parent, "scoped");
// 50 lines of code that might throw
sub.close();   // skipped on exception
```

### 4. Name every manager and every stored array

Unnamed managers show up as `NDManager(uid-142003136424600)` in the debug dump,
which is useless when you are looking for a leak. The same is true for
unnamed NDArrays. Always pass a descriptive name to `subMgr` and
`setName`.

### 5. Run `MemoryLeakTest` before merging

The test runs every agent end-to-end and asserts that the system-wide
resource count is stable. A leak that adds even a few arrays per step
will be caught after a single run of the test.

## Patterns

### Pattern: `INetwork.forward` with `@Cleanup` on the output

```java
default NDArray forward(NDArray input, final UnaryOperator<NDArray> block) {
    return scoped(it -> {
        @Cleanup var out = forward(it);                // close after block
        out.tempAttach(it.getManager());
        return block.apply(out);
    }, input);
}
```

The `@Cleanup` on `out` is the entire fix. Without it, every forward
leaks one tensor into the network manager.

### Pattern: noisy layer with scoped parameters

```java
@Cleanup var sub = subMgr(inputs.getManager(), "noisy-layer");
var input = inputs.singletonOrThrow();
var wMu = paramStore.getValue(weightMu, device, training);
var wSigma = paramStore.getValue(weightSigma, device, training);
var bMu = paramStore.getValue(biasMu, device, training);
var bSigma = paramStore.getValue(biasSigma, device, training);
sub.tempAttachAll(wMu, wSigma, bMu, bSigma);

if (training) {
    int inFeatures = (int) input.getShape().getLastDimension();
    ensureNoiseIsSampled(input.getManager(), inFeatures, outFeatures);
    var w = wMu.add(wSigma.mul(noise.epsWeight()));
    var b = bMu.add(bSigma.mul(noise.epsBias()));
    return Linear.linear(input, w, b);
}

return Linear.linear(input, wMu, bMu);
```

### Pattern: replay buffer store

```java
@Override
public void store(Experience exp) {
    exp.state().attach(subManager);          // permanent — lives in buffer
    exp.nextState().attach(subManager);
    // ... store, possibly close oldExp ...
}
```

The buffer's `subManager` is long-lived, and the stored states are
also long-lived (they will only be released when the buffer itself
closes). Permanent `attach` is the right call here.

### Pattern: sampled batch sub-manager

```java
@Override
public VecExperience sample(int batchSize) {
    if (!enough(batchSize)) return null;
    var sub = subMgr(subManager, "buffer-sample");   // named sub of buffer
    // ... build states, actions, ... inside `sub` ...
    return new VecExperience(sub, states, actions, rewards, nextStates, dones);
}
```

The `VecExperience` is responsible for closing `sub`; the caller uses
`@Cleanup var samples = replayBuffer.sample(...)`.

## Anti-patterns (the bugs we already fixed)

### A1. `tempAttach` inside `newAttachedList`

In `IReplayBuffer.newAttachedList`, the original code did:

```java
var mapped = mapper.apply(exp);
mapped.tempAttach(subManager);     // BAD
arrays.add(mapped.expandDims(0));
```

`tempAttach` restores `mapped` to its original manager (the replay
buffer's main sub) when `subManager` closes. Since the buffer's main
sub is long-lived, `mapped` is **restored and never closed**, leaking
one duplicate per experience per batch per sample. With `batchSize=32`
and two `newAttachedList` calls per sample, that is 64 leaks per step.

The fix: use `attach` (permanent), so `mapped` is owned by the
short-lived sample sub and closed when the sample sub closes.

### A2. `output.attach(modelManager)` in `safeForwardSingle`

`safeForwardSingle` was attaching every forward output to the model
manager, causing two leaks per training step (one for the online
forward, one for the target forward).

The fix lives in the caller (`INetwork.forward`), which now does
`@Cleanup var out = forward(it)`. The model manager is no longer
involved in the forward output's lifecycle.

### A3. `tempAttach` of batched duplicates that are then expanded

This is a special case of A1 but worth highlighting: any time you
`duplicate()` an array and then `expandDims` it, the duplicate itself
should be closed (it is no longer needed). The current pattern works
because `tempAttach` causes the duplicate to be restored to the buffer
manager, but that "restored" array is the leak — a permanent ghost
array that the buffer never explicitly closes.

## Adding a new component

Use this checklist when adding a new network, buffer, or agent.

- [ ] **Managers are named.** Every `newSubManager()` becomes
      `subMgr(parent, "descriptive-name")`.
- [ ] **Models are named.** Every `Model.newInstance(name, device)`
      becomes `newModel(getClass(), device)`.
- [ ] **AutoCloseable.** The component implements `AutoCloseable` and
      its `close()` closes every sub-manager it owns.
- [ ] **Scoping is explicit.** Any `NDArray` used only inside a block
      is created in a `@Cleanup var sub = subMgr(...)` scope.
- [ ] **Parameters are `tempAttach`-ed.** Long-lived parameters used in
      a scoped computation are `tempAttach`-ed to the scope's sub.
- [ ] **No permanent `attach` to the model manager.** Forward outputs
      and intermediates live in scoped subs, not in the model manager.
- [ ] **`@Cleanup` on forward outputs.** `INetwork.forward` callers
      declare `out` with `@Cleanup`.
- [ ] **Test added.** A new agent has a corresponding entry in
      `MemoryLeakTest`.

## Debugging a memory leak

When a leak is suspected, follow these steps in order.

### 1. Confirm the leak exists

Run the agent end-to-end and compare
`DJLMemoryManagement.systemResourceCount()` before and after:

```java
int before = DJLMemoryManagement.systemResourceCount();
runAgent();
int after = DJLMemoryManagement.systemResourceCount();
System.out.println("Leaked: " + (after - before));
```

If `after - before` is close to zero (under ~50 for a short run), no
leak. Otherwise, continue.

### 2. Locate the leaking manager

Add `DJLMemoryManagement.debugDump(parent)` at the end of the
training loop. The output shows every sub-manager and the count of
its resources, recursively:

```
\--- NDManager[parent-... | uid=...] resources=2 {PtNDManager=2}
    \--- NDManager[DeepQNetworkMLP-... | uid=...] resources=14 {PtNDArray=14}
    \--- NDManager[ExperienceReplayBuffer-... | uid=...] resources=1248488 {PtNDArray=1248488}
```

The sub-manager with the unexpectedly large `PtNDArray` count is your
leak source. The name (set via `subMgr(...)`) tells you exactly which
class created it.

### 3. Categorize the leak

- **Sub-manager count is non-zero and growing** — you are creating
  sub-managers somewhere that never close. Search for `newSubManager`
  and `subMgr` calls that are not paired with `@Cleanup` or
  try-with-resources.
- **PtNDArray count is non-zero and growing** — NDArrays are being
  attached to a long-lived manager. Look for `attach` calls that
  should be `tempAttach`, or for arrays that should be closed but
  aren't.
- **The count grows linearly with steps** — the leak is per-step.
  Look at the inner loop of `train()`: every `state.duplicate()`,
  every forward call, every `targetNet.forward` is a candidate.

### 4. Use `cap()` to localize the leak

DJL provides `NDManager.cap()` to block further attachments. Call it
on the manager you suspect, then run a single training step. If the
step throws, the offending `attach` call is in the code path that ran
right before the exception.

```java
parent.cap();   // throw on attach
agent.trainOneStep();
```

### 5. Confirm the fix

After applying a fix, re-run the agent and verify the count is stable
across multiple training steps. A single run is not enough — leaks
often scale with step count.

## Testing

The `MemoryLeakTest` class runs every agent's `main()` and asserts
that the system resource count does not grow by more than
`MAX_LEAK_PER_AGENT` (50) per agent.

To run:

```bash
mvn test -Dtest=MemoryLeakTest
```

A failure looks like:

```
AgentNoisyNetDQN leaked 1120 NDArrays after a full run (max allowed: 50).
Add DJLMemoryManagement.debugDump(NDManager.getSystemManager()) at the end
of the agent's train() method to identify the sub-manager that is
accumulating arrays.
```

The error message names the agent, the leaked count, and the debugging
steps to follow.

### When to add a new test

Whenever you add a new agent, new buffer, or new forward path,
**add a corresponding test method to `MemoryLeakTest`**. The test is
cheap (a few seconds per agent) and is the only thing standing
between future refactors and a re-introduction of the leaks we just
spent days hunting down.

## Reference: the bugs this guide prevents

For historical context, the leaks fixed while writing this guide were:

| File | Symptom | Root cause | Fix |
|------|---------|------------|-----|
| `IReplayBuffer.java:53` | 1.2M leaked arrays in buffer | `tempAttach` restoring duplicates to the buffer | Use `attach` |
| `safeForwardSingle.java:130` | Output of every forward pinned in the model manager | `output.attach(manager)` | Remove attach, let caller `@Cleanup` |
| `NoisyLayer.forwardInternal` | `w`, `b` and mul/add intermediates accumulating in the model manager | Binary ops in the parameter's manager | `tempAttach` parameters to a scoped sub |
| `FactorizedNoise.sampleNoiseOuter` | `fepsIn` and reshapes never closed | Created in `manager`, not in a sub | Allocate inside a scoped sub, `sub.ret` the keepers |
| `ExperienceReplayBuffer.sample` | Unnamed sub-manager | `newSubManager()` without `setName` | `subMgr(subManager, "buffer-sample")` |
| `PrioritizedReplayBuffer.sample` | Unnamed sub-manager | Same | `subMgr(subManager, "prioritized-sample")` |
| `AbstractAgent.run` | Unnamed sub-manager | Same | `subMgr(parent, "run-episode")` |
| `safeForwardSingle` | Unnamed sub-manager | Same | `subMgr(manager, "safe-forward-single")` |

The pattern is the same every time: **create in a long-lived manager
when you should have created in a short-lived one**. The helpers and
rules in this guide are designed to make the right thing easier than
the wrong thing.
