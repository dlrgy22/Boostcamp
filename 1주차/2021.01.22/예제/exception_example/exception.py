a = range(9)
for i in range(10):
    try:
        print(i, 10//i)
        print(a[i])
    except ZeroDivisionError as e:
        print("Not Divied 0")
    except Exception as e:
        print(e)
