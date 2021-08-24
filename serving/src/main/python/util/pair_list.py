class PairList(object):

    def __init__(self, keys=None, values=None, pair_list=None, pair_map=None):
        if keys and values:
            if len(keys) != len(values):
                raise ValueError("key value size mismatch.")
            self.keys = keys
            self.values = values
        elif pair_list:
            for pair in pair_list:
                self.keys.append(pair[0])
                self.values.append(pair[1])
        elif pair_map:
            for key, value in pair_map.items():
                self.keys.append(key)
                self.values.append(value)
        else:
            self.keys = []
            self.values = []

    def add(self, key=None, value=None, index=None, pair=None):
        if index and key and value:
            self.keys.insert(index, key)
            self.values.insert(index, value)
        elif pair:
            self.keys.append(pair[0])
            self.values.append(pair[1])
        elif key and value:
            self.keys.append(key)
            self.keys.append(value)

    def add_all(self, other):
        if other:
            self.keys.extend(other.keys())
            self.values.extend(other.values())

    def size(self):
        return len(self.keys)

    def is_empty(self):
        return self.size() == 0

    def get(self, index=None, key=None):
        if index:
            return self.keys[0], self.values[index]
        elif key:
            if key not in self.keys:
                return None
            key_index = self.keys.index(key)
            return self.values[key_index]

    def key_at(self, index: int):
        return self.keys.index(index)

    def value_at(self, index: int):
        return self.values.index(index)

    def get_keys(self):
        return self.keys

    def get_values(self) -> list:
        return self.values
