# Deep Learning Workflows

Deep learning workflows are a tool to help support endpoints requiring multiple models joined together. The workflows support a simple JSON/YAML configuration based language to help describe how to combine the models and even custom Java functions together.

## Examples

Here are a few examples of what such workflows might look like. The first example is a simple workflow involving pre-processing and post-processing:


```
-> PreProcess -> Model -> PostProcess ->
```


The next example is models used in sequence. Specifically, we can take the example of pose estimation which tries to find the joints in a human image which can then be used by applications like movie CGI. It first uses an object detection model to find the human and then the pose estimation to find the joints on it:


```
-> HumanDetection -> PoseEstimation ->
```


The final motivating example is using a model ensemble. In an ensemble, instead of training one model to solve the problem, it trains multiple and combines their results. This works similarly to having multiple people coming together to make a decision and often gives an easy accuracy boost:


```
-> PreProcess -> {Model1, Model2, Model3} -> Aggregate -> PostProcess ->
```


In summary, this tools focuses on specifying workflows involving multiple steps and multiple models. Steps can include both models and raw code. They also are running in both sequence and parallel.


## Workflow Design

### Global Configuration

As the system is built in YAML, the overall structure is a configuration object with options similar to:

```
# required
name: "MyWorkflow"
version: "1.2.0"

# Default model properties based on https://github.com/pytorch/serve/blob/master/docs/workflows.md#workflow-model-properties
# optional
minWorkers: 1
maxWorkers: 4
batchSize: 3
maxBatchDelayMillis: 5000
retryAttempts: 3
timeout: 5000

# Defined below
models: ...
functions: ...
workflow: ...
```

There are a few categories of configuration that are supported. The first is a basic string name and version to differentiate workflows.

After that are some performance properties. These will override the system-wide defaults, but can be further overridden in individual models.


### Models

The models section is used to declare the models that will be used within the workflow. It works such as an external definition as the models are built and trained separately.

Each model will be identified by a local model name. So, the overall section would look like:


```
models:
  modelA: ...
  modelB: ...
  modelC: ...
  ...
```


Each model individually would be an object describing how to load the model. The simplest case is where the file can be loaded just from a URL. One example is the [MMS .mar file](https://github.com/awslabs/multi-model-server) or through the DJL model zoo:


```
models:
  modelA: "https://example.com/path/to/model.mar"
  modelB: "djl://ai.djl.mxnet/ssd/0.0.1/ssd_512_resnet50_v1_voc"
```


A more advanced case can use an object representing the DJL criteria:


```
models:
  resnet:
    application: "cv/image_classification"
    engine: "MXNet"
    groupId: "ai.djl.mxnet"
    artifactId: "resnet"
    name: "Resnet"
    translator: "com.package.TranslatorClass"
    ...
```



### External Code

Along with the models, it is useful to be able to import functions written in Java. This can be used for custom preProcessing and postProcessing code along with other glue code necessary to combine the models together.

In order to use the functions, they must first be added to the Java classpath. Then, describe the class as follows:


```
functions:
  aggregate: "com.package.Aggregator"
```


The `Aggregator` would be a class that is required to have a public no-argument constructor and implement the `WorkflowFunction` interface. It might look something like:


```
public final class Aggregator implements ServingFunction {

  @Override
  public abstract CompletableFuture<Input> run(
            Workflow.WorkflowExecutor executor, List<Workflow.WorkflowArgument> args) {
    ...
  }
}
```


Here, the run function's arguments include the the `WorkflowExecutor` which can be used to execute higher models in the case of higher order functions. It also includes the `WorkflowArguments` which are the inputs to the function. More information can be found in the Javadoc of these two classes.

In addition, there are some number of built-in workflow functions as well. Right now, there is only the `IdentityWF`, but there are more to come.


### Workflow

Once all of the external definitions are defined, the actual workflow definition can be created. The workflow consists of a number of value definitions that can be thought of like final/immutable local variables in a function.

There are two important special values: `in` and `out`. The `in` is not defined, but must be used to refer to the input passed into the workflow. The `out` represents the output of the workflow and must always be defined.

Each definition consists of the result of a function application. Function application is written using an array where the first element of the array is the function/model name and the remaining elements are the arguments. This is similar to LISP. While LISP style functions are not the most common, it is chosen due to the constraints of fitting the definition into JSON/YAML.

Here is an example of the simple example given at the top of including a model, preprocessing, and postprocessing:


```
workflow:
  # First applies preProcessing to the input
  preProcessed: ["preProcess", "in"]
  
  # Then applies "model" to the result of preProcessing stored in the value "preProcessed"
  inferenced: ["model", "preProcessed"]
  
  # The output saved in the keyword "out" is done by applying postProcessing to the inferenced result
  out: ["postProcess", "inferenced"]
```


It is also possible to nest function calls by replacing arguments with a list. That means that this operation can be defined on a single line:


```
workflow:
  out: ["postProcess", ["model", ["preProcess", "in"]]]
```


To represent parallel operations, the data can be split. Here, each of the three models uses the same result of preProcessing so it is only computed once. Both “preProcess” and “postProcess” are just standard custom functions. Then, the results from the models are aggregated using the custom "aggregate" function and passed to postProcessing. It would also be possible to combine "postProcessing" and "aggregate" or to use a predefined function in place of "aggregate":


```
workflow:
  preProcessed: ["preProcess", "in"]
  m1: ["model1", "preProcessed"]
  m2: ["model2", "preProcessed"]
  m3: ["model3", "preProcessed"]
  out: ["postProcess", ["aggregate", "m1", "m2", "m3"]]
```


As a final example, here is one that features a more complicated interaction. The human detection model will find all of the humans in an image. Then, the "splitHumans" function will turn all of them into separate images that can be treated as a list. The "map" will apply the "poseEstimation" model to each of the detected humans in the list.


```
workflow:
  humans: ["splitHumans", ["humanDetection", "in"]]
  out: ["map", "poseEstimation", "humans"]
```
