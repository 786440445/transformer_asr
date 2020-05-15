import os
from collections import Counter


def is_chinese(c):
    return c >= '\u4E00' and c <= '\u9FA5'


if __name__ == '__main__':

    file = './train_chars.txt' #多
    file2 = './train_chars1.txt' #少
    counter = Counter([])
    for path, _, files in os.walk('../corpus'):
        for file in files:
            if file[-3:] == 'txt':
                print(file)
                with open(os.path.join(path, file)) as f:
                    for line in f.readlines():
                        line = line.strip('\n').split('\t')[2]
                        counter.update(list(line))
    data = ''
    print(counter.items())
    print(counter)
    items = sorted(counter)
    for i, key in enumerate(items):
        if is_chinese(key):
            data += key + '\n'
    data = data[:-1]
    with open('./train_chars.txt', 'w') as f:
        f.writelines(data)