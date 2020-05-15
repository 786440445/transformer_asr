import jieba
from src.dataset import AudioDataset

def main():
    aishell_dataset = AudioDataset('test', 1)
    han_list = aishell_dataset.han_lst
    dict = {}
    for data in han_list:
        print(data)
        words = jieba.lcut(data)
        for word in words:
            if len(word) == 1:
                continue
            else:
                dict[word] = dict.get(word, 0) + 1

    items = list(dict.items())
    items.sort(key=lambda x: x[1], reverse=True)
    wirte_lines = ""
    with open('../../tmp/keywords.txt', 'w+') as f:
        for i in range(len(items)):
            word, count = items[i]
            wirte_lines += word + ' ' + str(count) + '\n'
        f.writelines(wirte_lines[:-1])


if __name__ == '__main__':
    # main()
    print(jieba.lcut('违规以租代征改变规划条件等用地一万公顷'))