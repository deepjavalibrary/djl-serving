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

import os
import pickle
import mmap

from djl_python import Input
from collections import OrderedDict


class SessionManager:

    def __init__(self, properties: dict):
        self.limit = properties.get("option.sessions.limit", None)
        if self.limit is not None:
            self.limit = int(self.limit)

    def load(self, session_id):
        """
    Loads the data associated with a session_id
    :param session_id: the id for the session to load
    :return: loads the data (what kind of data depend on what type of SessionManager)
    """
        pass

    def save(self, session_id, value):
        """
    Saves the data associated with a session_id
    :param session_id: the id for the session to save
    :param value: the updated data to save for the session
    :return: None
    """
        pass

    def remove(self, session_id):
        """
    Removes the data associated with a session_id
    :param session_id: the id for the session to remove
    :return: None
    """
        pass

    def prune(self, limit, backup=None):
        """
    Prunes all data, leaving only the newest limit sessions remaining
    :param limit: how many data items to leave
    :param backup: an optional SessionManager to backup the pruned data to (todo)
    :return: None
    """
        pass


class LocalSessionManager(SessionManager):

    def __init__(self, properties: dict):
        super().__init__(properties)
        self.sessions: dict = OrderedDict()

    def load(self, session_id):
        return self.sessions.get(session_id, None)

    def save(self, session_id, value):
        self.sessions[session_id] = value
        self.sessions.move_to_end(session_id)
        if self.limit is not None and len(self.sessions) > self.limit:
            self.prune(self.limit)

    def remove(self, session_id):
        return self.sessions.pop(session_id)

    def prune(self, limit, backup: SessionManager = None):
        while len(self.sessions) > self.limit:
            popped_id, popped_value = self.sessions.popitem(0)
            if backup is not None:
                backup.save(popped_id, popped_value)


class FileSessionManager(SessionManager):

    def __init__(self, properties: dict):
        super().__init__(properties)
        self.files_path = properties["option.sessions.path"]

    def load(self, session_id):
        path = self._path(session_id)
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, session_id, value):
        with open(self._path(session_id), "wb") as f:
            pickle.dump(value, f)
        if self.limit is not None:
            self.prune(self.limit)

    def remove(self, session_id):
        os.remove(self._path(session_id))

    def prune(self, limit, backup: SessionManager = None):
        files = list(os.listdir(self.files_path))
        if len(files) < limit:
            return
        files.sort(key=lambda f: os.path.getmtime(self._path(f)))
        to_prune = files[:len(files) - limit]
        for session_id in to_prune:
            if backup is not None:
                backup.save(session_id, self._path(session_id))
            os.remove(self._path(session_id))

    def _path(self, session_id):
        return os.path.join(self.files_path, session_id.replace("/", "-"))


class MmapSessionManager(FileSessionManager):

    def __init__(self, properties: dict):
        super().__init__(properties)
        self.file_size = int(properties["option.sessions.file_size"])
        self.opened = {}

    def load(self, session_id):
        path = self._path(session_id)
        new_mmap = not os.path.isfile(path)
        if new_mmap:
            with open(path, "wb") as f:
                f.write(self.file_size * b'\x00')
        f = open(path, "r+")
        m = mmap.mmap(f.fileno(),
                      length=self.file_size,
                      access=mmap.ACCESS_WRITE)
        self.opened[session_id] = (f, m)
        return m, new_mmap

    def save(self, session_id, value=None):
        if session_id in self.opened:
            f, m = self.opened[session_id]
            m.close()
            f.close()
        if self.limit is not None:
            self.prune(self.limit)


def get_session_manager(properties: dict):
    if "option.sessions" not in properties:
        return None
    mode = properties["option.sessions"].lower()
    if mode == "local":
        return LocalSessionManager(properties)
    elif mode == "files":
        return FileSessionManager(properties)
    elif mode == "mmap":
        return MmapSessionManager(properties)
    else:
        raise ValueError("Unknown session manager type: " + mode)
