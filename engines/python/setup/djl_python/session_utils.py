#!/usr/bin/env python
#
# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import logging

from datetime import datetime, timezone
from djl_python.async_utils import create_non_stream_output
from djl_python.outputs import Output

logger = logging.getLogger(__name__)

HEADER_SAGEMAKER_SESSION_ID = "X-Amzn-SageMaker-Session-Id"
HEADER_SAGEMAKER_CLOSED_SESSION_ID = "X-Amzn-SageMaker-Closed-Session-Id"


async def create_session(request):
    session_manager, inputs = request
    try:
        session = session_manager.create_session()
        expiration_ts = datetime.fromtimestamp(
            session.expiration_ts,
            tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info(f"Session {session.session_id} created")
        return {
            "data": {
                "result": f"Session {session.session_id} created"
            },
            "properties": {
                HEADER_SAGEMAKER_SESSION_ID:
                f"{session.session_id}; Expires={expiration_ts}"
            }
        }
    except Exception as e:
        return {"error": f"Failed to create session: {str(e)}", "code": 424}


async def close_session(request):
    session_manager, inputs = request
    session_id = inputs.get_property(HEADER_SAGEMAKER_SESSION_ID)
    try:
        session_manager.close_session(session_id)
        logger.info(f"Session {session_id} closed")
        return {
            "data": {
                "result": f"Session {session_id} closed"
            },
            "properties": {
                HEADER_SAGEMAKER_CLOSED_SESSION_ID: f"{session_id}"
            }
        }
    except Exception as e:
        return {"error": f"Failed to close session: {str(e)}", "code": 424}


def get_session(session_manager, request):
    session_id = request.get_property(HEADER_SAGEMAKER_SESSION_ID)
    if session_manager is None:
        if session_id is not None:
            raise RuntimeError(
                f"invalid payload. stateful sessions not enabled, {HEADER_SAGEMAKER_SESSION_ID} header not supported"
            )
        return None
    session = session_manager.get_session(session_id)
    return session


def session_non_stream_output_formatter(
    response: dict,
    **_,
) -> Output:
    if "error" in response:
        return create_non_stream_output("",
                                        error=response["error"],
                                        code=response["code"])

    return create_non_stream_output(response["data"],
                                    properties=response.get("properties"))
