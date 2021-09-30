# DJL Serving - WorkLoadManager

The djl-serving serving can be divided into a frontend and backend.
The frontend is a [netty](https://netty.io/) webserver that manages incoming requests and operators the control plane.
The backend WorkLoadManager handles the model batching, workers, and threading for high-performance inference.

For those who already have a web server infrastructure but want to operate high-performance inference, it is possible to use only the WorkLoadManager.
For this reason, we have it split apart into a separate module.

Using the WorkLoadManager is quite simple. First, create a new one through the constructor:

```java
WorkLoadManager wlm = new WorkLoadManager();
```

You can also configure the WorkLoadManager by using the static `WlmConfigManager`.

Then, you can construct an instance of the `ModelInfo` for each model you will want to run through `wlm`.
With the `ModelInfo`, you are able to build a `Job` once you receive input:

```java
ModelInfo modelInfo = new ModelInfo(...);
Job job = new Job(modelInfo, input);
```

Once you have your job, it can be submitted to the WorkLoadManager.
It will automatically spin up workers if none are created and manage worker numbers.
Then, it returns a `CompletableFuture<Output>` for the result.

```java
CompletableFuture<Output> futureResult = wlm.runJob(job);
```
