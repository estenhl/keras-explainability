from typing import Any, Dict


class LRPStrategy:
    def __init__(self, layers: Dict[str, Any]):
        self.layers = layers
