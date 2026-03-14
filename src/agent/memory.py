from collections import deque


class ShortTermMemory:

    def __init__(self, max_size=20):
        self.memory = deque(maxlen=max_size)

    def add(self, entry):
        self.memory.append(entry)

    def get_recent(self):
        return list(self.memory)