class MetaCompany(type):
    def __init__(cls, name, bases, dict):
        super().__init__(name, bases, dict)
        cls.is_meta = True