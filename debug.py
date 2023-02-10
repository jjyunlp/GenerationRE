from pydantic.main import BaseModel

class A():
    aa = "aa"
    def __init__(self, a):
        self.a = a
        print(a)
        self.F(a+" we invoke this class method in init")
    
    @classmethod
    def F(self, b):
        print(b)


class B(BaseModel):
    b = "This is a class variable, default"
    @classmethod
    def print_class_variable(cls,):
        print("In print_class_variable")

    @classmethod
    def class_function(cls, input):
        cls.print_class_variable()
        return cls(b=input)

def function(a, b):
    print(a, b)

if __name__ == "__main__":
    a = A("This is a instance of class A")
    A.F("Can we invoke a function in class A without initialization?")
    print(a.aa)
    b = B(b="This is b, a class variable")
    b = B()
    b.print_class_variable()
    a = {"a": [111], "b": [222]}
    function(**a)
    for v in a.values():
        print(type(v))
    
    out = B.class_function("bb")
    
    print(type(B.class_function("bb")))
    print(type(B()))
    a = []
    b = [1,2]
    for i in a:
        for j in b:
            print(i, j)
            print("i,j")