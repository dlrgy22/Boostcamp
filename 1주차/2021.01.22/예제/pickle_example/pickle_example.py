# 예제
import pickle

# test를 binary 파일로 저장
f = open("list.pickle", "wb")
test = [1, 2, 3, 4, 5]
pickle.dump(test, f)
f.close()

# 저장되어있는 list를 load  
load_test = pickle.load(f)
print(load_test)
f.close()