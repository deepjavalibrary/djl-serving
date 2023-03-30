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
package ai.djl.serving.plugins.ddb;

import ai.djl.modality.Output;
import ai.djl.serving.cache.CacheEngine;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import software.amazon.awssdk.core.SdkBytes;
import software.amazon.awssdk.services.dynamodb.DynamoDbClient;
import software.amazon.awssdk.services.dynamodb.model.AttributeValue;
import software.amazon.awssdk.services.dynamodb.model.CreateTableRequest;
import software.amazon.awssdk.services.dynamodb.model.DeleteItemRequest;
import software.amazon.awssdk.services.dynamodb.model.DescribeTableRequest;
import software.amazon.awssdk.services.dynamodb.model.GetItemRequest;
import software.amazon.awssdk.services.dynamodb.model.GetItemResponse;
import software.amazon.awssdk.services.dynamodb.model.KeySchemaElement;
import software.amazon.awssdk.services.dynamodb.model.KeyType;
import software.amazon.awssdk.services.dynamodb.model.PutItemRequest;
import software.amazon.awssdk.services.dynamodb.model.ResourceNotFoundException;
import software.amazon.awssdk.services.dynamodb.waiters.DynamoDbWaiter;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/** A {@link CacheEngine} that stores elements in DynamoDB. */
public class DDBCacheEngine implements CacheEngine {

    private static final String KEY_NAME = "CACHE_ID";
    private static final String CONTENT_NAME = "CONTENT";
    private static final String TABLE_NAME = "djl-serving-pagination-table";
    private static final Logger logger = LoggerFactory.getLogger(DDBCacheEngine.class);
    private DynamoDbClient ddbClient;

    /** Constructs a {@link DDBCacheEngine}. */
    public DDBCacheEngine() {
        ddbClient = DynamoDbClient.create();
        try {
            ddbClient.describeTable(DescribeTableRequest.builder().tableName(TABLE_NAME).build());
        } catch (ResourceNotFoundException e) {
            logger.info("Dynamo db table {} doesn't exist, attempting to create....", TABLE_NAME);
            ddbClient.createTable(
                    CreateTableRequest.builder()
                            .tableName(TABLE_NAME)
                            .keySchema(
                                    KeySchemaElement.builder()
                                            .attributeName(KEY_NAME)
                                            .keyType(KeyType.HASH)
                                            .build())
                            .build());
            try (DynamoDbWaiter waiter = DynamoDbWaiter.builder().client(ddbClient).build()) {
                waiter.waitUntilTableExists(
                        DescribeTableRequest.builder().tableName(TABLE_NAME).build());
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean isMultiTenant() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public void put(String key, Output output) {
        CompletableFuture.supplyAsync(
                () -> {
                    Map<String, AttributeValue> map = new ConcurrentHashMap<>();
                    try {
                        map.put(KEY_NAME, AttributeValue.fromS(key));
                        map.put(
                                CONTENT_NAME,
                                AttributeValue.fromB(SdkBytes.fromByteArray(output.encode())));
                        ddbClient.putItem(
                                PutItemRequest.builder().tableName(TABLE_NAME).item(map).build());
                    } catch (IOException e) {
                        throw new IllegalArgumentException(
                                "Output object could not be encoded!", e);
                    }
                    return 0;
                });
    }

    /** {@inheritDoc} */
    @Override
    public Output get(String key) {
        Map<String, AttributeValue> map = new ConcurrentHashMap<>();
        map.put(KEY_NAME, AttributeValue.fromS(key));
        GetItemResponse response =
                ddbClient.getItem(GetItemRequest.builder().tableName(TABLE_NAME).key(map).build());
        try (InputStream is = response.item().get(CONTENT_NAME).b().asInputStream()) {
            return Output.decode(is);
        } catch (IOException e) {
            throw new IllegalArgumentException("Output object could not be decoded!", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void remove(String key) {
        CompletableFuture.supplyAsync(
                () -> {
                    Map<String, AttributeValue> map = new ConcurrentHashMap<>();
                    map.put(KEY_NAME, AttributeValue.fromS(key));
                    ddbClient.deleteItem(
                            DeleteItemRequest.builder().tableName(TABLE_NAME).key(map).build());
                    return 0;
                });
    }
}
