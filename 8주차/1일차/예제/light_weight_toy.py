import sys
a = 256
print(a == 256)
print(a is 256)
b = 257
print(b == 257)
print(b is 257)

# is => 주소, == 값


print(sys.getsizeof(0))
print(sys.getsizeof(1))
print(sys.getsizeof(2**30 - 1))
print(sys.getsizeof(2**30))
print(sys.getsizeof(2**60 -1))
print(sys.getsizeof(2**60))
print(sys.getsizeof(2**90 -1))
print(sys.getsizeof(2**90))
print(sys.int_info)

