# Workflow Templates

Workflow templates are a tool to make it easier to register similar workflows.

## Registering a workflow with a template

To register a new workflow using a template, the template must first be registered (see below).
Then, you can use the management API to register the workflow.
To do so, call register while specifying the template using the `template` query parameter.
Then, specify all template replacement values using additional query parameters.

## Available templates

### adapter

The adapter template is used for a simple workflow that reflects an [adapter model](adapters.md).

Parameters:

- template=`adapter`
- adapter - The adapter name
- url - The adapter URL
- model - The model URL

Example:

`POST /workflows?template=adapter&adapter={adapterName}&url={adapterUrl}&model={modelUrl}`

## Adding new templates

To add a new template, begin by creating the template JSON file.
This mostly matches the standard format of a [workflow](workflows.md).
However, your template can indicate variable sections of the template to be replaced.
This is done by surrounding the name with curly braces.
So, a parameter `param` would replace the value `{param}` within the template.
This replacement is directly in place, so if your parameter is a string you will still have to surround it with quotation marks.

Then, you must register your new workflow template.
There are two options for doing this.
First, you can add it to the classpath as a resource with path `workflowTemplates/{workflowTemplateName}.json`.
Alternatively, you can register it from a plugin by calling `WorkflowTemplates.register(..)`.
Once this is done, you will be able to begin creating workflows with your template.