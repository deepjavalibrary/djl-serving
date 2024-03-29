# Global Configuration

This covers configurations that are used globally and as part of startup for DJL Serving.

## Command line parameters

User can use the following parameters to start djl-serving, those parameters will override default behavior:

```
djl-serving -h

usage: djl-serving [OPTIONS]
 -f,--config-file <CONFIG-FILE>    Path to the configuration properties file.
 -h,--help                         Print this help.
 -m,--models <MODELS>              Models to be loaded at startup.
 -s,--model-store <MODELS-STORE>   Model store location where models can be loaded.
```

Details about the models, model-store, and workflows can be found in the equivalent configuration properties.

## config.properties file

DJL Serving use a `config.properties` file to store configurations.

### Configure listening port

DJL Serving only allows localhost access by default.

* inference_address: inference API binding address, default: http://127.0.0.1:8080
* management_address: management API binding address, default: http://127.0.0.1:8081

Here are a couple of examples:

```properties
# bind inference API to all network interfaces with SSL enabled
inference_address=https://0.0.0.0:8443

# bind inference API to private network interfaces
inference_address=https://172.16.1.10:8443
```

### Configure initial models and workflows

**Model Store**

The `model_store` config property can be used to define a directory where each file/folder in it is a model to be loaded.
It will then attempt to load all of them by default.
Here is an example:

```properties
model_store=build/models
```

**Load Models**

The `load_models` config property can be used to define a list of models (or workflows) to be loaded.
The list should be defined as a comma separated list of urls to load models from.

Each model can be defined either as a URL directly or optionally with prepended endpoint data like `[EndpointData]=modelUrl`.
The endpoint is a list of data items separated by commas.
The possible variations are:

- `[modelName]`
- `[modelName:version]`
- `[modelName:version:engine]`
- `[modelName:version:engine:deviceNames]`

The version can be an arbitrary string.
The engines uses the standard DJL `Engine` names.

Possible deviceNames strings include `*` for all devices and a `;` separated list of device names following the format defined in DJL `Device.fromName`.
If no device is specified, it will use the DJL default device (usually GPU if available else CPU).

```properties
load_models=djl://ai.djl.zoo/mlp,[mlp:v1:PyTorch:*]=https://resources.djl.ai/test-models/mlp.zip
```

**Workflows**

Use the `load_models` config property to define initial workflows that should be loaded on startup.

```properties
load_models=https://resources.djl.ai/test-models/basic-serving-workflow.json
```

View the [workflow documentation](workflows.md) to see more information about workflows and their configuration format.

### Enable SSL

For users who want to enable HTTPs, you can change `inference_address` or `management_addrss`
protocol from http to https, for example: `inference_addrss=https://127.0.0.1`.
This will make DJL Serving listen on localhost 443 port to accepting https request.

User also must provide certificate and private keys to enable SSL. DJL Serving support two ways to configure SSL:

1. Use keystore
    * keystore: Keystore file location, if multiple private key entry in the keystore, first one will be picked.
    * keystore_pass: keystore password, key password (if applicable) MUST be the same as keystore password.
    * keystore_type: type of keystore, default: PKCS12

2. Use private-key/certificate files
    * private_key_file: private key file location, support both PKCS8 and OpenSSL private key.
    * certificate_file: X509 certificate chain file location.

#### Self-signed certificate example

This is a quick example to enable SSL with self-signed certificate

##### User java keytool to create keystore

```bash
keytool -genkey -keyalg RSA -alias djl -keystore keystore.p12 -storepass changeit -storetype PKCS12 -validity 3600 -keysize 2048 -dname "CN=www.MY_DOMSON.com, OU=Cloud Service, O=model server, L=Palo Alto, ST=California, C=US"
```

Config following property in config.properties:

```properties
inference_address=https://127.0.0.1:8443
management_address=https://127.0.0.1:8444
keystore=keystore.p12
keystore_pass=changeit
keystore_type=PKCS12
```

##### User OpenSSL to create private key and certificate

```bash
# generate a private key with the correct length
openssl genrsa -out private-key.pem 2048

# generate corresponding public key
openssl rsa -in private-key.pem -pubout -out public-key.pem

# create a self-signed certificate
openssl req -new -x509 -key private-key.pem -out cert.pem -days 360

# convert pem to pfx/p12 keystore
openssl pkcs12 -export -inkey private-key.pem -in cert.pem -out keystore.p12
```

Config following property in config.properties:

```properties
inference_address=https://127.0.0.1:8443
management_address=https://127.0.0.1:8444
keystore=keystore.p12
keystore_pass=changeit
keystore_type=PKCS12
```

## Environment variables

User can set environment variables to change DJL Serving behavior, following is a list of
variables that user can be set for DJL Serving:

* JAVA_HOME
* JAVA_OPTS
* SERVING_OPTS
* MODEL_SERVER_HOME

**Note:** environment variable has higher priority that command line or config.properties.
It will override other property values.

### Global Model Server settings

Global settings are configured at model server level. Change to these settings usually requires
restart model server to take effect.

Most of the model server specific configuration can be configured in `conf/config.properties` file.
You can find the configuration keys here:
[ConfigManager.java](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/src/main/java/ai/djl/serving/util/ConfigManager.java#L52-L79)

Each configuration key can also be override by environment variable with `SERVING_` prefix, for example:

```
export SERVING_JOB_QUEUE_SIZE=1000 # This will override JOB_QUEUE_SIZE in the config
```

| Key               | Type    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|-------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MODEL_SERVER_HOME | env var | DJLServing home directory, default: Installation directory (e.g. /usr/local/Cellar/djl-serving/0.27.0/)                                                                                                                                                                                                                                                                                                                                                   |
| DEFAULT_JVM_OPTS  | env var | default: `-Dlog4j.configurationFile=${APP_HOME}/conf/log4j2.xml`<br>Override default JVM startup options and system properties.                                                                                                                                                                                                                                                                                                                           |
| JAVA_OPTS         | env var | default: `-Xms1g -Xmx1g -XX:+ExitOnOutOfMemoryError`<br>Add extra JVM options.                                                                                                                                                                                                                                                                                                                                                                            |
| SERVING_OPTS      | env var | default: N/A<br>Add serving related JVM options.<br>Some of DJL configuration can only be configured by JVM system properties, user has to set DEFAULT_JVM_OPTS environment variable to configure them.<br>- `-Dai.djl.pytorch.num_interop_threads=2`, this will override interop threads for PyTorch<br>- `-Dai.djl.pytorch.num_threads=2`, this will override OMP_NUM_THREADS for PyTorch<br>- `-Dai.djl.logging.level=debug` change DJL loggging level |


## Appendix

### How to configure logging

#### Option 1: enable debug log:

```
export SERVING_OPTS="-Dai.djl.logging.level=debug"
```

#### Option 2: use your log4j2.xml

```
export DEFAULT_JVM_OPTS="-Dlog4j.configurationFile=/MY_CONF/log4j2.xml
```

DJLServing provides a few built-in `log4j2-XXX.xml` files in DJLServing containers.
Use the following environment variable to print HTTP access log to console:

```
export DEFAULT_JVM_OPTS="-Dlog4j.configurationFile=/usr/local/djl-serving-0.27.0/conf/log4j2-access.xml
```

Use the following environment variable to print both access log, server metrics and model metrics to console:

```
export DEFAULT_JVM_OPTS="-Dlog4j.configurationFile=/usr/local/djl-serving-0.27.0/conf/log4j2-console.xml
```

