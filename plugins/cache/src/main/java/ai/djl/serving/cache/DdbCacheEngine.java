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
package ai.djl.serving.cache;

import ai.djl.modality.Output;
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.AwsCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.SdkBytes;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dynamodb.DynamoDbClient;
import software.amazon.awssdk.services.dynamodb.model.AttributeDefinition;
import software.amazon.awssdk.services.dynamodb.model.AttributeValue;
import software.amazon.awssdk.services.dynamodb.model.BatchWriteItemRequest;
import software.amazon.awssdk.services.dynamodb.model.BillingMode;
import software.amazon.awssdk.services.dynamodb.model.CreateTableRequest;
import software.amazon.awssdk.services.dynamodb.model.DeleteRequest;
import software.amazon.awssdk.services.dynamodb.model.DescribeTableRequest;
import software.amazon.awssdk.services.dynamodb.model.KeySchemaElement;
import software.amazon.awssdk.services.dynamodb.model.KeyType;
import software.amazon.awssdk.services.dynamodb.model.PutItemRequest;
import software.amazon.awssdk.services.dynamodb.model.QueryRequest;
import software.amazon.awssdk.services.dynamodb.model.QueryResponse;
import software.amazon.awssdk.services.dynamodb.model.ResourceNotFoundException;
import software.amazon.awssdk.services.dynamodb.model.ScalarAttributeType;
import software.amazon.awssdk.services.dynamodb.model.TimeToLiveSpecification;
import software.amazon.awssdk.services.dynamodb.model.UpdateTimeToLiveRequest;
import software.amazon.awssdk.services.dynamodb.model.WriteRequest;
import software.amazon.awssdk.services.dynamodb.waiters.DynamoDbWaiter;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A {@link CacheEngine} that stores elements in DynamoDB. */
public final class DdbCacheEngine extends BaseCacheEngine {

    private static final Logger logger = LoggerFactory.getLogger(DdbCacheEngine.class);

    private static final String TABLE_NAME =
            Utils.getenv("SERVING_DDB_TABLE_NAME", "djl-serving-pagination-table");
    private static final String CACHE_ID = "CACHE_ID";
    private static final String INDEX = "INDEX_KEY";
    private static final String HEADER = "HEADER";
    private static final String CONTENT = "CONTENT";
    private static final String LAST_CONTENT = "LAST_CONTENT";
    private static final String TTL = "TTL";
    private static final String EXPRESSION =
            CACHE_ID + "=:" + CACHE_ID + " and " + INDEX + ">=:" + INDEX;

    private DynamoDbClient ddbClient;
    private long cacheTtl;

    /**
     * Constructs a {@link DdbCacheEngine}.
     *
     * @param ddbClient DynamoDB client
     */
    private DdbCacheEngine(DynamoDbClient ddbClient) {
        this.ddbClient = ddbClient;
        cacheTtl = Duration.ofMillis(30).toMillis();
        writeBatch = Integer.parseInt(Utils.getenv("SERVING_CACHE_BATCH", "5"));
    }

    /**
     * Creates a new instance.
     *
     * @return a new {@code DdbCacheEngine} instance
     */
    public static DdbCacheEngine newInstance() {
        DynamoDbClient ddbClient;
        String endpoint = System.getProperty("DDB_ENDPOINT");
        if (endpoint == null) {
            ddbClient = DynamoDbClient.create();
        } else {
            // For local unit test only
            URI uri = URI.create(endpoint);
            AwsCredentials credentials = AwsBasicCredentials.create("fake", "key");
            ddbClient =
                    DynamoDbClient.builder()
                            .region(Region.US_EAST_1)
                            .endpointOverride(uri)
                            .credentialsProvider(StaticCredentialsProvider.create(credentials))
                            .build();
        }
        try {
            ddbClient.describeTable(DescribeTableRequest.builder().tableName(TABLE_NAME).build());
            return new DdbCacheEngine(ddbClient);
        } catch (ResourceNotFoundException e) {
            logger.info("DynamoDB table {} doesn't exist, attempting to create....", TABLE_NAME);
            CreateTableRequest request =
                    CreateTableRequest.builder()
                            .tableName(TABLE_NAME)
                            .attributeDefinitions(
                                    AttributeDefinition.builder()
                                            .attributeName(CACHE_ID)
                                            .attributeType(ScalarAttributeType.S)
                                            .build(),
                                    AttributeDefinition.builder()
                                            .attributeName(INDEX)
                                            .attributeType(ScalarAttributeType.N)
                                            .build())
                            .keySchema(
                                    KeySchemaElement.builder()
                                            .attributeName(CACHE_ID)
                                            .keyType(KeyType.HASH)
                                            .build(),
                                    KeySchemaElement.builder()
                                            .attributeName(INDEX)
                                            .keyType(KeyType.RANGE)
                                            .build())
                            .billingMode(BillingMode.PAY_PER_REQUEST)
                            .build();
            try (DynamoDbWaiter waiter = DynamoDbWaiter.builder().client(ddbClient).build()) {
                ddbClient.createTable(request);
                waiter.waitUntilTableExists(
                        DescribeTableRequest.builder().tableName(TABLE_NAME).build());
                TimeToLiveSpecification spec =
                        TimeToLiveSpecification.builder().attributeName(TTL).enabled(true).build();
                UpdateTimeToLiveRequest req =
                        UpdateTimeToLiveRequest.builder()
                                .tableName(TABLE_NAME)
                                .timeToLiveSpecification(spec)
                                .build();
                ddbClient.updateTimeToLive(req);
                return new DdbCacheEngine(ddbClient);
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean isMultiTenant() {
        return false;
    }

    @Override
    protected void putSingle(String key, Output output, boolean last) {
        String ttl = String.valueOf(System.currentTimeMillis() + cacheTtl);
        writeDdb(key, output, null, -1, ttl, last);
    }

    @Override
    protected void putStream(String key, Output output, byte[] buf, int index, boolean last) {
        String ttl = String.valueOf(System.currentTimeMillis() + cacheTtl);
        writeDdb(key, output, buf, index, ttl, last);
    }

    /** {@inheritDoc} */
    @Override
    public Output get(String key, int start, int limit) {
        int shiftedStart = start == 0 ? -1 : start;
        Map<String, AttributeValue> attrValues = new ConcurrentHashMap<>();
        attrValues.put(':' + CACHE_ID, AttributeValue.builder().s(key).build());
        attrValues.put(
                ':' + INDEX, AttributeValue.builder().n(String.valueOf(shiftedStart)).build());
        QueryRequest request =
                QueryRequest.builder()
                        .tableName(TABLE_NAME)
                        .keyConditionExpression(EXPRESSION)
                        .expressionAttributeValues(attrValues)
                        .limit(limit == Integer.MAX_VALUE ? limit : limit + 1)
                        .build();

        QueryResponse response = ddbClient.query(request);
        if (response.count() == 0) {
            return null;
        }

        Output output = new Output();
        boolean complete = false;
        boolean first = true;
        List<byte[]> list = new ArrayList<>();
        for (Map<String, AttributeValue> item : response.items()) {
            // skip first one
            if (first) {
                first = false;
                continue;
            }
            AttributeValue header = item.get(HEADER);
            if (header != null) {
                Output o = decode(header);
                output.setCode(o.getCode());
                output.setMessage(o.getMessage());
                output.setProperties(o.getProperties());
            }
            AttributeValue content = item.get(CONTENT);
            if (content != null) {
                list.add(content.b().asByteArrayUnsafe());
            }
            AttributeValue lastContent = item.get(LAST_CONTENT);
            if (lastContent != null) {
                complete = true;
            }
            start = Integer.parseInt(item.get(INDEX).n());
        }
        if (!list.isEmpty()) {
            output.add(joinBytes(list));
        }
        if (!complete) {
            String startString = start <= 0 ? "" : Integer.toString(start);
            output.addProperty("x-next-token", key + startString);
            output.addProperty(
                    "X-Amzn-SageMaker-Custom-Attributes", "x-next-token=" + key + startString);
        }
        return output;
    }

    /** {@inheritDoc} */
    @Override
    public void remove(String key) {
        Map<String, AttributeValue> attrValues = new ConcurrentHashMap<>();
        attrValues.put(':' + CACHE_ID, AttributeValue.builder().s(key).build());

        QueryRequest query =
                QueryRequest.builder()
                        .tableName(TABLE_NAME)
                        .keyConditionExpression(CACHE_ID + " = :" + CACHE_ID)
                        .expressionAttributeValues(attrValues)
                        .projectionExpression(INDEX)
                        .build();
        QueryResponse response = ddbClient.query(query);

        List<WriteRequest> list = new ArrayList<>();
        for (Map<String, AttributeValue> item : response.items()) {
            AttributeValue index = item.get(INDEX);
            Map<String, AttributeValue> map = new ConcurrentHashMap<>();
            map.put(CACHE_ID, AttributeValue.builder().s(key).build());
            map.put(INDEX, AttributeValue.builder().n(index.n()).build());
            DeleteRequest req = DeleteRequest.builder().key(map).build();
            list.add(WriteRequest.builder().deleteRequest(req).build());
        }
        BatchWriteItemRequest batch =
                BatchWriteItemRequest.builder().requestItems(Map.of(TABLE_NAME, list)).build();
        ddbClient.batchWriteItem(batch);
    }

    private Output decode(AttributeValue header) {
        byte[] buf = header.b().asByteArrayUnsafe();
        try {
            return Output.decode(new ByteArrayInputStream(buf));
        } catch (IOException e) {
            throw new AssertionError("Decode output failed.", e);
        }
    }

    void writeDdb(String key, Output output, byte[] buf, int index, String cacheTtl, boolean last) {
        Map<String, AttributeValue> map = new ConcurrentHashMap<>();
        try {
            map.put(CACHE_ID, AttributeValue.fromS(key));
            map.put(INDEX, AttributeValue.fromN(String.valueOf(index)));
            map.put(TTL, AttributeValue.fromN(cacheTtl));
            if (output != null) {
                map.put(HEADER, AttributeValue.fromB(SdkBytes.fromByteArray(output.encode())));
            }
            if (buf != null) {
                map.put(CONTENT, AttributeValue.fromB(SdkBytes.fromByteArray(buf)));
            }
            if (last) {
                map.put(LAST_CONTENT, AttributeValue.fromBool(Boolean.TRUE));
            }
            ddbClient.putItem(PutItemRequest.builder().tableName(TABLE_NAME).item(map).build());
        } catch (IOException e) {
            throw new AssertionError("Output object could not be encoded!", e);
        }
    }
}
