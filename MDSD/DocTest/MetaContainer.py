class MetaContainer(type):
    def __init__(cls,name,parent,dct):
        super(MetaContainer, cls).__init__(name,parent,dct)
        cls.has_name = "MetaContainer"
        pass