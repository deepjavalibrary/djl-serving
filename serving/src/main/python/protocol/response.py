class Response(object):
    def __init__(self):
        self.len = None
        self.content = None

    def get_len(self):
        return self.len

    def set_len(self, len):
        self.len = len

    def get_buffer_arr(self):
        return self.content

    def set_buffer_arr(self, content):
        self.content = content
