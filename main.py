# 자연어 처리
from konlpy.tag import Mecab

def tokenize_sentense(text):
    mecab = Mecab()
    return mecab.morphs(text)


data = ''
with open('./data/data', 'r') as f:
    data = f.read()

data = data.split('\n')
for i in range(len(data)):
    data[i] = [int(data[i][0]), data[i][2:]]

train = data[0:40]
test = data[40:54]

