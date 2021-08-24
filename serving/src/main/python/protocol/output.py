class Output(object):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        self.request_id = None
        self.properties = dict()
        self.content = None

    def set_request_id(self, request_id):
        self.request_id = request_id

    def get_request_id(self):
        return self.request_id

    def set_properties(self, properties):
        self.properties = properties

    def get_properties(self):
        return self.properties

    def set_content(self):
        return self.content
