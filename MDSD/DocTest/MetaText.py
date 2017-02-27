class MetaText(type):
    def __init__(cls,name,parent,dct):
        super(MetaText, cls).__init__(name,parent,dict)
        cls.has_name = "MetaText"