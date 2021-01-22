# 하나의 지문에 단어가 몇개가 있을까?
from collections import defaultdict
from collections import Counter

text ="""A press release is the quickest and easiest way to get free publicity. 
        If well written, a press release can result in multiple published articles about your firm and its products. 
        And that can mean new prospects contacting you asking you to sell to them. ….""".lower().split()
        
def get_word_count():
    word_count = defaultdict(lambda: 0)
    for word in text:
        word_count[word] += 1

    for element in sorted(word_count.items(), key = lambda x: x[1], reverse = True):
        yield element[0], element[1]

gw = get_word_count()
for i in gw:
    print(i)
