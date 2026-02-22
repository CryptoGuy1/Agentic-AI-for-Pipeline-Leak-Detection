from collections import deque

class ShortTermMemory:
    def __init__(self, max_size=10):
        self.memory = deque(maxlen=max_size)

    def add(self, state, action, explanation):
        entry = {
            "state": state,
            "action": action,
            "explanation": explanation
        }
        self.memory.append(entry)

    def get_recent(self):
        return list(self.memory)

    def clear(self):
        self.memory.clear()