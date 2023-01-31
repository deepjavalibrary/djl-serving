/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.ndarray.NDList;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;

public class BinaryClient {

    private static final URI testUrl = URI.create("http://127.0.0.1:8080/predictions/test");
    private static final HttpClient httpClient =
            HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1).build();

    public static NDList postNDList(NDList list) throws IOException, InterruptedException {
        HttpRequest request =
                HttpRequest.newBuilder()
                        .POST(BodyPublishers.ofByteArray(list.encode()))
                        .uri(testUrl)
                        .setHeader("content-type", "tensor/ndlist")
                        .build();
        HttpResponse<byte[]> response = httpClient.send(request, BodyHandlers.ofByteArray());
        return NDList.decode(list.getManager(), response.body());
    }
}
