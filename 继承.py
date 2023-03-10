class animal():
    def __init__(self,name):
        self.name = name
    def greet(self):
        print('i am ' + self.name)


A = animal('Dog')
A.greet()


class hello():
    def __init__(self):
        self.a, self.b = 0, 1
    def __iter__(self):
        return self   # iter
    def __next__(self):
        self.a, self.b = self.b,self.a + self.b
        return self.a
H  = hello()
for i in H:  # 类返回的值是 self.a
    if i > 10:
        break
    print(i)


class maxindex():
    def __init__(self,max_index):
        self.index = -1
        self.max_index = max_index
    def __iter__(self):
        return self       #  @summary: 迭代器，生成迭代对象时调用，返回值必须是对象自己,然后for可以循环调用next方法
    def __next__(self):   #  @summary: 每一次for循环都调用该方法（必须存在）
        self.index += 1
        if self.index >= self.max_index:
            raise StopIteration
        return self.index

index = maxindex(3)
for i in index:
    print(i)

