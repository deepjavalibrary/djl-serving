# Security Policy

## How we do security

As much as possible, DJL Serving relies on automated tools to do security scanning. In particular, we support:

1. Docker CVE patch scanning: Using AWS ECR
2. Dependency Analysis: Using Dependabot
3. Code Analysis: Using CodeQL

## Important Security Guidelines

1. DJL Serving has two APIs: inference and management. More can be available through plugins.
    HTTP - `8080` for both the inference and management API

    By default, both APIs are available on port `8080` through HTTP and accessible to `localhost`.  The addresses can be configured by following the guide for
    [global configuration](serving/docs/configurations_global.md).
    DJL Serving does not prevent users from configuring the address to be of any value, including the wildcard address `0.0.0.0`.
    Please be aware of the security risks of configuring the address to be `0.0.0.0` as this will give all addresses (including publicly accessible addresses)
    on the host, access to the DJL Serving endpoints listening on the ports shown above.
    You should be especially careful with the management API including setting it to a publicly accessible address or forwarding it's port to one.
    If expose, it could allow an attacker to create models and execute arbitrary code on your machine.
2. By [default](serving/docker/Dockerfile), the docker images are configured to expose the port `8080` to the host.
   The default DJL Serving configuration in the container, which is executed by the docker entrypoint, will expose both the inference and management APIs set to `http://0.0.0.0:8080`.
   This is designed for internal isolated services or development work. For other use cases, provide alternative configurations to avoid exposing the management API.

3. Be sure to validate the authenticity of all model files and model artifacts being used with DJL Serving.
    1. A model file being downloaded from the internet from an untrustworthy source may have malicious code. This can compromise the integrity of your application, slow your device, or extract data.
       Data that is on the device may include configurations, model files, inference requests, logs, and other standard important data such as security keys.
       1. For the common case of python models with our default handler and models from HuggingFace IDs, we require an additional option `options.trust_remote_code` to enable custom python files.
       2. Remember that an attacker may still use other kinds of models, so this precaution alone will not ensure security.
    2. DJL Serving executes the arbitrary python code packaged in the model file. Make sure that you've either audited that the code you're using is safe and/or is from a source that you trust.
    3. DJL Serving supports custom [plugins](plugins). These should also only be used from trusted sources.
    4. Running DJL Serving inside a container environment and loading an untrusted model file does not guarantee isolation from a security perspective.
    5. It is possible for models to specify additional files to download. This includes options such as the `option.model_id` which can download from HuggingFace Hub or S3 and providing a `requirements.txt` file which can download from [PyPI](https://pypi.org/) and other URLs.
       When using these or other features that enable additional downloads from your model, you must also ensure the authenticity and security of the resources being downloaded by your model.
4. Enable SSL:

    DJL Serving supports two ways to configure SSL:
    1. Using a keystore
    2. Using private-key/certificate files

    You can find more details in the [configuration guide](serving/docs/configurations_global.md#enable-ssl).
5. Prepare your model against bad inputs and prompt injections. Some recommendations:
    1. Pre-analysis: check how the model performs by default when exposed to prompt injection (e.g. using [fuzzing for prompt injection](https://github.com/FonduAI/awesome-prompt-injection?tab=readme-ov-file#tools)).
    2. Input Sanitation: Before feeding data to the model, sanitize inputs rigorously. This involves techniques such as:
        - Validation: Enforce strict rules on allowed characters and data types.
        - Filtering: Remove potentially malicious scripts or code fragments.
        - Encoding: Convert special characters into safe representations.
        - Verification: Run tooling that identifies potential script injections (e.g. [models that detect prompt injection attempts](https://python.langchain.com/docs/guides/safety/hugging_face_prompt_injection)).
6. If you intend to run multiple models in parallel with shared memory, it is your responsibility to ensure the models do not interact or access each other's data. The primary areas of concern are tenant isolation, resource allocation, model sharing and hardware attacks.
7. There are various options and settings within DJL Serving that can be controlled through the use of environment variables.
   This includes configurations at [all levels of the DJL Serving stack](serving/docs/configuration.md), so ensure that no malicious environment variables can be passed to DJL Serving.

## Reporting a Vulnerability

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.
