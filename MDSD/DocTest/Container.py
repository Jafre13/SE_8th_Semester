from MetaContainer import *

class Container(metaclass=MetaContainer):
    pass


print(Container.has_name)