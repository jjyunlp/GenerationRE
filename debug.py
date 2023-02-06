class A():
    def __init__(self, a):
        self.a = a
        print(a)
        self.F(a+" we invoke this class method in init")
    
    @classmethod
    def F(self, b):
        print(b)

if __name__ == "__main__":
    a = A("This is a instance of class A")
    A.F("Can we invoke a function in class A without initialization?")