/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.awscurl;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.net.URI;
import java.util.Date;

public class AwsV4SignerTest {

    @Test
    public void testSigner() {
        String service = "sagemaker";
        AwsCredentials creds = new AwsCredentials("id", "key", null);
        AwsV4Signer signer = new AwsV4Signer(service, "us-east-1", creds);
        signer.setOverrideDate(new Date());
        Assert.assertNotNull(signer.getOverriddenDate());

        URI uri = URI.create("https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/");
        SignableRequest request = new SignableRequest(service, uri);
        request = request.copy();

        signer.presignRequest(request, creds, new Date());
    }
}
