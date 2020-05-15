import pandas as pd


class LoadData():
    def __init__(self, data_args, type):
        self.data_type = type
        self.thchs30 = data_args.thchs30
        self.aishell = data_args.aishell
        self.stcmd = data_args.stcmd
        self.aidatatang = data_args.aidatatang
        self.magic_data = data_args.magic_data
        self.aidatatang_1505 = data_args.aidatatang_1505
        self.prime = data_args.prime

        self.bgn_data = data_args.bgn_data
        self.echo_data = data_args.echo_data
        self.noise_data = data_args.noise_data
        self.rate_data = data_args.rate_data

        self.train_length = data_args.train_length
        self.path_lst = []
        self.pny_lst = []
        self.han_lst = []
        self.source_init()

    def source_init(self):
        """
        txt文件初始化，加载
        :return:
        """
        print('get source list...')
        read_files = []
        if self.data_type == 'train':
            if self.thchs30 == True:
                read_files.append('thchs_train.txt')
            if self.aishell == True:
                read_files.append('aishell_train.txt')
            if self.stcmd == True:
                read_files.append('stcmd_train.txt')
            if self.aidatatang == True:
                read_files.append('aidatatang_train.txt')
            if self.magic_data == True:
                read_files.append('magicdata_train.txt')
            if self.aidatatang_1505 == True:
                read_files.append('aidatatang_1505_train.txt')
            if self.prime == True:
                read_files.append('prime.txt')
            if self.bgn_data == True:
                read_files.append('noise_corpus/1505_train_bgn.txt')
            if self.echo_data == True:
                read_files.append('noise_corpus/1505_train_echo.txt')
            if self.noise_data == True:
                read_files.append('noise_corpus/1505_train_noise.txt')
            if self.rate_data == True:
                read_files.append('noise_corpus/1505_train_rate.txt')

        elif self.data_type == 'dev':
            if self.thchs30 == True:
                read_files.append('thchs_dev.txt')
            if self.aishell == True:
                read_files.append('aishell_dev.txt')
            if self.stcmd == True:
                read_files.append('stcmd_dev.txt')
            if self.aidatatang == True:
                read_files.append('aidatatang_dev.txt')
            if self.magic_data == True:
                read_files.append('magicdata_dev.txt')
            if self.aidatatang_1505 == True:
                read_files.append('aidatatang_1505_dev.txt')

        elif self.data_type == 'test':
            if self.thchs30 == True:
                read_files.append('thchs_test.txt')
            if self.aishell == True:
                read_files.append('aishell_test.txt')
            if self.stcmd == True:
                read_files.append('stcmd_test.txt')
            if self.aidatatang == True:
                read_files.append('aidatatang_test.txt')
            if self.magic_data == True:
                read_files.append('magicdata_test.txt')
            if self.aidatatang_1505 == True:
                read_files.append('aidatatang_1505_test.txt')

        for file in read_files:
            print('load ', file, ' data...')
            sub_file = './corpus/' + file
            data = pd.read_table(sub_file, header=None)
            paths = data.iloc[:, 0].tolist()
            pny = data.iloc[:, 1].tolist()
            hanzi = data.iloc[:, 2].tolist()
            self.path_lst.extend(paths)
            self.pny_lst.extend(pny)
            self.han_lst.extend(hanzi)
        if self.train_length:
            self.path_lst = self.path_lst[:self.train_length]
            self.pny_lst = self.pny_lst[:self.train_length]
            self.han_lst = self.han_lst[:self.train_length]