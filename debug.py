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
    def print_class_variable(self,):
        print(b)
        print(self.b)
        print(b)

def function(a, b):
    print(a, b)

if __name__ == "__main__":
    a = A("This is a instance of class A")
    A.F("Can we invoke a function in class A without initialization?")
    print(a.aa)
    b = B(b="This is b, a class variable")
    b = B()
    b.print_class_variable()
    a = {"a": 111, "b": 222}
    function(**a)