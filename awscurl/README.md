# awscurl

This tool provided a [curl](https://curl.haxx.se/docs/manpage.html) like API to make request to AWS services.

Some AWS API (e.g. aws sagemaker-runtime) arbitrary request body.
It's hard to hand craft these body. This tools uses curl compatible
command line options to help construct different type of request body.

## Downloading awscurl

You can download awscurl like this:

```sh
wget https://publish.djl.ai/awscurl/awscurl \
&& chmod +x awscurl
```

## Building From Source

You can build it using gradle:

```sh
./gradlew build
```
You will find a jar file in build/libs folder.

## Usage

### AWS Credentials
This tool uses the same ways as AWS CLI to load AWS credentials:
[Configuring the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html)

#### Using environment variables

#### Using java System properties

#### Using profile
You can use --profile option to use [Named Profiles](https://docs.aws.amazon.com/cli/latest/userguide/cli-multiple-profiles.html).

### Obtain region
region name is requird to sign SigV4 request. region can be obtained in following order:

1. passed by parameter --region
2. inferred from url if url using \[XXX.\]SERVICE.REGION-NAME.amazonaws.com or \[XXX.\]REGION-NAME.SERVICE.amazonaws.com format.
3. get region name from ~/.aws/credentials file
4. get region name from ~/.aws/config file

### Show help

```sh
java -jar awscurl.jar -h
```

### Example

```sh
awscurl -n sagemaker https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/mms-demo/invocations -F "data=@kitten.jpg" -F "model_name=squeezenet_v1.1"
```

## Limitation
* Doesn't support [SigV4 Chunked upload](https://docs.aws.amazon.com/AmazonS3/latest/API/sigv4-streaming.html)
* This tool load request body into memory, the max size of body can up to 2G. The actual body is also limited by java heap size.
* Unlike AWS CLI can find URL by itself, caller must specify AWS service endpoint url.
