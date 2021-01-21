class Product(object):
    pass

class Inventory(object):
    def __init__(self):
        self.__items= []  # Private 변수로 선언 타객체가 접근하지 못함

    @property
    def items(self):
        return self.__items[:]  #복사해서 넣어주기 때문에 변화 X

    def add_new_item(self, product):
        if type(product) == Product:
            self.__items.append(product)
            print("newitemadded")
        else:
            raise ValueError("InvalidItem")

    def get_number_of_items(self):
        return len(self.__items)

my_inventory = Inventory()
my_inventory.add_new_item(Product())
my_inventory.add_new_item(Product())
get_item =my_inventory.items
print(my_inventory.get_number_of_items())