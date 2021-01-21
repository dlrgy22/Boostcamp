class Person(object): # 부모클래스 Person 선언
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender

    def about_me(self): # Method 선언
        print("저의이름은", self.name, "이구요, 제나이는", str(self.age), "살입니다.")

class Employee(Person): # 부모클래스Person으로부터상속
    def __init__(self, name, age, gender, salary, hire_date):
        super().__init__(name, age, gender) # 부모객체사용
        self.salary= salary
        self.hire_date= hire_date# 속성값추가

    def do_work(self): # 새로운메서드추가
        print("열심히일을합니다.")

    def about_me(self): # 부모클래스함수재정의
        super().about_me() # 부모클래스함수사용
        print("제급여는", self.salary, "원이구요, 제입사일은", self.hire_date, " 입니다.")

me = Employee("ikhyo", 25, 'male', "100", "2021.01.21")
me.about_me()
print(me.name)
