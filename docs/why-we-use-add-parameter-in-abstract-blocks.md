# Why We Use `addParameter` in `AbstractBlock`

> A short note on what `addParameter(...)` actually does in DJL, and why
> "just put the field in the constructor" silently breaks a custom block.

## The question this doc answers

When you write a custom layer that extends `AbstractBlock`, DJL forces you
to declare every trainable weight as a `Parameter` and register it through
`addParameter(...)`. It is tempting to skip that and keep a plain
`NDArray weightMu;` field instead, on the grounds that:

1. It is less code.
2. Java fields are just as easy to set in the constructor.
3. The forward pass still works — autograd can backprop into any tensor
   with `requires_grad = true`.

The catch is that `addParameter` is not a bookkeeping hint. It is the
single line that plugs a tensor into five different subsystems of DJL.
Skip it and four of those subsystems never see your weights.

The rest of this document walks through the five consumers of the
`parameters` map and shows what each one does to your block — and what
stops working if the field is not registered.

## What `addParameter` actually does

`AbstractBlock.java:81, 142-145` is the entire mechanism:

```java
// AbstractBlock.java:81
protected LinkedHashMap<String, Parameter> parameters = new LinkedHashMap<>();

// AbstractBlock.java:142-145
protected final <P extends Parameter> P addParameter(P parameter) {
    parameters.put(parameter.getName(), parameter);
    return parameter;
}
```

That is the whole of it. Everything interesting happens *because* the
parameter is in this `LinkedHashMap`. The five consumers below all read
it back via `getDirectParameters()` (`AbstractBlock.java:159-161`).

## The five subsystems that read the map

### 1. Initialization — your `Initializer` is applied here

`AbstractBaseBlock.java:183-186`:

```java
for (Parameter parameter : getDirectParameters().values()) {
    // Attach arrays to params if params are null; set require gradient if required
    parameter.initialize(manager, dataType);
}
```

This is the only place DJL ever calls `parameter.initialize(...)`. If
your custom layer does something like

```java
this.weightMu = addParameter(Parameter.builder()
        .setName("weightMu")
        .setType(Parameter.Type.WEIGHT)
        .build());
this.weightMu.setInitializer(NoisyLayerInit.ofMu(inFeatures));
```

the `setInitializer` call stores an `Initializer` reference on the
`Parameter` object. Nothing happens to the actual tensor values until the
loop above runs `parameter.initialize(...)`. Without the registration,
the loop never iterates over your `weightMu` and its tensor stays
`null`/uninitialized — the next forward pass will throw.

### 2. Save / load — yes, exactly the symptom you suspected

`AbstractBaseBlock.java:275-298`:

```java
public void saveParameters(DataOutputStream os) throws IOException {
    os.write(version);
    saveMetadata(os);
    for (Parameter parameter : getDirectParameters().values()) {
        parameter.save(os);
    }
    for (Block child : getChildren().values()) {
        child.saveParameters(os);
    }
}

public void loadParameters(NDManager manager, DataInputStream is)
        throws IOException, MalformedModelException {
    byte loadVersion = is.readByte();
    loadMetadata(loadVersion, is);
    for (Parameter parameter : getDirectParameters().values()) {
        parameter.load(manager, is);
    }
    for (Block child : getChildren().values()) {
        child.loadParameters(manager, is);
    }
}
```

These two methods are called by `BaseModel.save` and `BaseModel.load`
respectively:

- `BaseModel.java:305` → `block.saveParameters(dos);`
- `BaseModel.java:401` → `block.loadParameters(manager, dis);`

Both methods iterate `getDirectParameters().values()`. An unregistered
field is invisible here. Consequences:

- **Save**: nothing about your layer's weights is written to the
  checkpoint. The file size shrinks and the saved model is incomplete.
- **Load**: nothing is restored from the checkpoint. The freshly
  constructed `NDArray` you put in the field at construction time is
  what the model uses — random init, or stale values, or `null` if
  the field was uninitialized.

This is the failure mode that is most visible and most often blamed on
`addParameter`. It is real, but it is the *least* dangerous of the
five.

### 3. The optimizer — gradients are computed but never applied

This is the silent killer. `Trainer.java:206-214`:

```java
public void step() {
    if (!gradientsChecked) {
        checkGradients();
    }
    long begin = System.nanoTime();
    parameterStore.updateAllParameters();
    addMetric("step", begin);
}
```

`ParameterStore.updateAllParameters` walks only the parameters that
have been requested via `getValue` during forward. In the standard flow,
that mirrors back to the registered parameters from `getParameters()`.
DJL's optimizer updates *only* the tensors it can see.

The kicker: PyTorch's autograd does not care who "owns" the tensor.
If your `weightMu` and `weightSigma` are plain `NDArray` fields with
`requires_grad = true`, autograd will happily compute gradients for
them during `backward()`. Those gradients will land on the tensors'
`gradient` slot and... nothing will ever read them, because the
optimizer's update step only iterates the registered parameters.

Symptom in practice: the loss might appear to decrease for a few steps
(if the random init was lucky) and then plateau or oscillate, because
no corrective update ever lands on the offending fields. There is no
exception, no log line, no warning. The model just quietly fails to
learn.

### 4. Gradient sanity checks

`Trainer.java:341-363` calls `checkGradients()` before the first
`step()` to confirm at least one parameter has a non-zero gradient:

```java
model.getBlock().getParameters().values().stream()
        .filter(Parameter::requiresGradient)
        .forEach(...)
```

If you skip `addParameter` for every field, the stream is empty and
the check throws:

```
IllegalStateException: Gradient values are all zeros, please call
gradientCollector.backward() on your target NDArray (usually loss),
before calling step()
```

That exception is misleading — it points you at the loss / backward
call, but the real cause is that the registered parameter set is empty.
Either way, you do get an early failure, which is the one piece of good
news.

### 5. Multi-device mirroring

`Trainer.java:118-124`:

```java
model.getBlock().initialize(model.getNDManager(), model.getDataType(), shapes);
model.getBlock()
        .getParameters()
        .forEach(
                pair -> {
                    for (Device device : devices) {
                        ...
                    }
                });
```

When you train with `setDevices(gpu0, gpu1)`, DJL creates a copy of
every registered parameter on each device and swaps them in and out
per forward pass. Unregistered fields stay on whatever device you
attached them to at construction. If you ever move from single-GPU or
CPU training to multi-GPU, an unregistered weight simply will not
follow — you will compute the forward on GPU tensors multiplied with
a CPU tensor and get a device-mismatch exception.

## The recursion contract: `getParameters` vs `getDirectParameters`

Worth highlighting because it surprises people:

`AbstractBaseBlock.java:234-244`:

```java
public ParameterList getParameters() {
    ParameterList allParams = getDirectParameters();
    for (Pair<String, Block> childPair : getChildren()) {
        for (Pair<String, Parameter> paramPair : childPair.getValue().getParameters()) {
            allParams.add(childPair.getKey() + "_" + paramPair.getKey(), paramPair.getValue());
        }
    }
    return allParams;
}
```

`getParameters()` is recursive — it walks child blocks and prefixes
their parameter names with the child block's name. `getDirectParameters()`
is only the parameters declared on `this` block. Most of the consumers
above call `getParameters()` (so the recursion does the work for them),
but the contract is the same: only what is in your `parameters` map
becomes part of the model's identity.

## Could I work around this by overriding `saveMetadata`?

You can dodge failure modes (1) and (2) by overriding
`saveMetadata` / `loadMetadata` and writing your own `NDArray` directly
to a `DataOutputStream`. That fixes persistence for the cost of a
hand-rolled serializer.

You cannot dodge failure modes (3), (4) and (5) without re-implementing
`Parameter` yourself. Those are not "save the bytes" problems — they
are "the trainer has no idea your field exists" problems. The only
clean fix is to use the contract DJL exposes for "this field is a
learnable, persistent, device-aware, optimizer-tracked tensor." That
contract is `addParameter`.

## A note on `SequentialBlock`

`SequentialBlock` does not replace `addParameter`. It is a container
whose `add(...)` / `addAll(...)` methods call `addChildBlock(name, block)`
for each child. The leaf blocks inside the sequence (your `NoisyLayer`,
`Linear`, etc.) still have to register their own parameters. You use
`SequentialBlock` for *composition* and `AbstractBlock` +
`addParameter` for any *layer that owns trainable weights*.

## TL;DR

`addParameter` is doing five jobs at once:

| Subsystem | What it does for you | Symptom if missing |
|---|---|---|
| Initialization | Applies the `Initializer` you set on the parameter | First forward throws "parameter not initialized" |
| Save | Writes the tensor to the checkpoint | Empty weights for that block in the saved file |
| Load | Reads the tensor back from the checkpoint | Stale / random values after a load round-trip |
| Optimizer step | Knows the parameter exists and applies gradients | Gradients computed but never applied — model fails to learn |
| Multi-device mirror | Copies the parameter onto every training device | Device-mismatch crash as soon as you use more than one device |

Skip it and you have to reimplement the entire `Parameter` class to
recover four of those five. The persistence one is the only one you
can patch around with `saveMetadata` / `loadMetadata`. Use
`addParameter`.
