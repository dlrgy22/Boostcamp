class Animal:
    def __init__(self, name): # Constructorof theclass
        self.name = name
    def talk(self): # Abstract method, defined by convention only
        raise NotImplementedError("Subclass must implement abstract method")   #자식 class에서 구현을 하지 않았을시 강제로 에러발생

class Cat(Animal):
    def talk(self):
        return'Meow!'

class Dog(Animal):
    def talk(self):
        return'Woof! Woof!'

animals= [Cat('Missy'),Cat('Mr. Mistoffelees'),Dog('Lassie')]

for animal in animals:
    print(animal.name + ': ' + animal.talk())