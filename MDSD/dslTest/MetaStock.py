class MetaStock(type):
    def __init__(cls, name, bases, dct):
        if hasattr(cls, "owned_by"):
            cls.owned_by = ""