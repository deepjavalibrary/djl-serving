#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import atexit
import logging
import os
import pickle
import re
import shutil
import tempfile
import time
import uuid
import subprocess as sp

import numpy as np

from djl_python.aws.cloud_watch import CloudWatch

UUID_PATTERN = re.compile("[0-9a-f-]{36}")


class Session:

    def __init__(self, session_id: str, session_root: str):
        self.session_id = session_id
        self.files_path = os.path.join(session_root, session_id)

    def put(self, key: str, value):
        with open(self._path(key), "wb") as f:
            pickle.dump(value, f)

    def get_as_numpy(self, key: str, shape, dtype=np.float32, create=False):
        if create:
            open(self._path(key), "wb").close()
        return np.memmap(self._path(key), dtype=dtype, mode="r+", shape=shape)

    def get(self, key: str, d=None):
        path = self._path(key)
        if not os.path.isfile(path):
            return d

        with open(path, "rb") as f:
            return pickle.load(f)

    def remove(self):
        if os.path.exists(self.files_path):
            logging.info(f"closing session: {self.session_id}")
            shutil.rmtree(self.files_path)
            return True
        else:
            logging.warning(f"session not found: {self.session_id}")
            return False

    def _path(self, key: str):
        return os.path.join(self.files_path, key.replace("/", "-"))


class SessionManager:

    def __init__(self, properties: dict):
        self.expiration = int(
            properties.get("sessions_expiration", str(20 * 60)))
        self.cloud_watch = CloudWatch()
        if os.path.exists("/dev/shm"):
            session_dir = "/dev/shm/djl_sessions"
        else:
            session_dir = os.path.join(tempfile.gettempdir(), "djl_sessions")

        self.sessions_path = properties.get("sessions_path", session_dir)
        self.sessions_s3url = properties.get("sessions_s3url", None)
        if not os.path.exists(self.sessions_path):
            os.makedirs(self.sessions_path)

        atexit.register(self._save_sessions_to_s3)

    def create_session(self) -> Session:
        """
        Creates a new session
        :return: the new session id
        """
        self._clean_expired_session()
        session_id = str(uuid.uuid4())
        session = Session(session_id, self.sessions_path)
        os.makedirs(session.files_path)
        session.put(".creation_time", time.time())

        self.cloud_watch.post("create_session")
        return session

    def get_session(self, session_id: str) -> (Session | None):
        if not session_id or not UUID_PATTERN.match(session_id):
            raise ValueError(f"invalid session_id: {session_id}")

        session = Session(session_id, self.sessions_path)
        if not os.path.exists(session.files_path):
            return self._recover_from_s3(session)

        return session

    def close_session(self, session_id):
        if not session_id or not UUID_PATTERN.match(session_id):
            raise ValueError(f"invalid session_id: {session_id}")

        session = Session(session_id, self.sessions_path)
        if session.remove():
            self.cloud_watch.post("close_session")

    def _clean_expired_session(self):
        sessions = os.listdir(self.sessions_path)
        for session_id in sessions:
            session = Session(session_id, self.sessions_path)
            if time.time() - session.get(".creation_time") > self.expiration:
                self.close_session(session_id)

    def _recover_from_s3(self, session: Session) -> (Session | None):
        if not self.sessions_s3url:
            return None

        logging.info(f"Restoring session {session.session_id} from s3...")
        os.makedirs(session.files_path)
        command = [
            "/opt/djl/bin/s5cmd", "--retry-count", "1", "sync",
            f"{self.sessions_s3url}/{session.session_id}/*",
            f"{session.files_path}"
        ]
        result = sp.run(command)
        if result.returncode == 0:
            return session

        logging.warning(f"s5cmd download failed: {result.stderr}")
        shutil.rmtree(session.files_path)
        return None

    def _save_sessions_to_s3(self):
        if not self.sessions_s3url:
            return None

        logging.info("Session manager shutdown, backup sessions to s3...")
        command = [
            "/opt/djl/bin/s5cmd", "--retry-count", "1", "sync",
            f"{self.sessions_path}/*", f"{self.sessions_s3url}/"
        ]
        result = sp.run(command)
        if result.returncode != 0:
            logging.warning(f"s5cmd upload failed: {result.stderr}")
            return None
        return None
