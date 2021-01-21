print("\n\n>>>> closer 예제")
def tag_func(tag, text):
    text = text
    tag = tag

    def inner_func():
        return'<{0}>{1}<{0}>'.format(tag, text)

    return inner_func

h1_func = tag_func('title', "This is Python Class")
p_func= tag_func('p', "Data Academy")
print(h1_func())
print(p_func())

################################################################



print("\n\n>>>> decorator 예제")
def star(func):
    def inner(*args, **kargs):
        print("*" * 30)
        func(*args, **kargs)
        print("*" * 30)
    
    return inner

def printer(msg):
    print(msg)

deco_star = star(printer)
#print(deco_star)
deco_star("hello")




################################################################

print("\n\n>>>> decorator 예제2")
def star(func):
    def inner(*args, **kargs):
        print("*" * 30)
        func(*args, **kargs)
        print("*" * 30)
    
    return inner

@star
def printer(msg):
    print(msg)
printer("hello")


################################################################
print("\n\n>>>> 2중 decorator 예제")
def star(func):
    def inner(*args, **kargs):
        print("*" * 30)
        func(*args, **kargs)
        print("*" * 30)
    
    return inner

def percent(func):
    def inner(*args, **kargs):
        print("%" * 30)
        func(*args, **kargs)
        print("%" * 30)
    
    return inner

@percent
@star   #안 부터 실행
def printer(msg):
        print(msg)
printer("hello")



