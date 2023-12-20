# DJL Serving - WorkLoadManager

DJL Serving can be divided into a frontend and backend.
The frontend is a [netty](https://netty.io/) webserver that manages incoming requests and operates the control plane.
The backend WorkLoadManager handles the model batching, workers, and threading for high-performance inference.

For those who already have a web server infrastructure but want to operate high-performance inference, it is possible to use only the WorkLoadManager.
For this reason, we have it split apart into a separate module.

Using the WorkLoadManager is quite simple. First, create a new one through the constructor:

```java
WorkLoadManager wlm = new WorkLoadManager();
```

You can also configure the WorkLoadManager by using the static [`WlmConfigManager`](https://javadoc.io/doc/ai.djl.serving/wlm/latest/ai/djl/serving/wlm/util/WlmConfigManager.html).

Then, you can construct a [`ModelInfo`](https://javadoc.io/doc/ai.djl.serving/wlm/latest/ai/djl/serving/wlm/ModelInfo.html) for each model you will want to run through `wlm`.
With the `ModelInfo`, you are able to build a [`Job`](https://javadoc.io/doc/ai.djl.serving/wlm/latest/ai/djl/serving/wlm/Job.html) once you receive input:

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

View the javadocs for the [`WorkLoadManager`](https://javadoc.io/doc/ai.djl.serving/wlm/latest/ai/djl/serving/wlm/WorkLoadManager.html) for more options.

## Documentation

The latest javadocs can be found on the [javadoc.io](https://javadoc.io/doc/ai.djl.serving/wlm/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.


## Installation
You can pull the server from the central Maven repository by including the following dependency:

```xml
<dependency>
    <groupId>ai.djl.serving</groupId>
    <artifactId>wlm</artifactId>
    <version>0.25.0</version>
</dependency>
```

