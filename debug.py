from pydantic.main import BaseModel
import torch.nn as nn
import torch

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
    
    t = torch.Tensor([1,2,3])
    print(t)
    act = nn.Tanh()
    t = act(t)
    print(act(t))
    t = act(t)
    print(act(t))

    a = [0.3, 0.28, 0.12, 0.1, 0.1, 0.1]
    T = 0.4 #临时给个值试试 
    # 将概率分布进行sharpen处理，需要是np array，list不行
    import numpy as np
    p = np.asarray(a)
    pt = p**(1/T)  # 0.5前后调整，若T=1，则相当于没有
    targets = pt / sum(pt)
    a = targets.tolist()
    print(a)