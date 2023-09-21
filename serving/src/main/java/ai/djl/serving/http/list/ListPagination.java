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
package ai.djl.serving.http.list;

import ai.djl.serving.util.NettyUtils;

import io.netty.handler.codec.http.QueryStringDecoder;

/** A pagination helper for items in the list responses. */
public final class ListPagination {

    private int pageToken;
    private int last;

    /**
     * Constructs a new {@link ListPagination}.
     *
     * @param decoder the query with the pagination data
     * @param keysSize the number of items to paginate over
     */
    public ListPagination(QueryStringDecoder decoder, int keysSize) {
        int limit = NettyUtils.getIntParameter(decoder, "limit", 100);
        pageToken = NettyUtils.getIntParameter(decoder, "next_page_token", 0);
        if (limit > 100 || limit < 0) {
            limit = 100;
        }
        if (pageToken < 0) {
            pageToken = 0;
        }

        last = pageToken + limit;
        if (last > keysSize) {
            last = keysSize;
        }
    }

    /**
     * Returns the current page token.
     *
     * @return the current page token
     */
    public int getPageToken() {
        return pageToken;
    }

    /**
     * Returns the last item in the pagination.
     *
     * @return the last item in the pagination
     */
    public int getLast() {
        return last;
    }
}
