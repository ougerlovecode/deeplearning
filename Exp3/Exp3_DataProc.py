"""
这个文件中可以添加数据预处理相关函数和类等
如词汇表生成，Word转ID（即词转下标）等
此文件为非必要部分，可以将该文件中的内容拆分进其他部分
"""
import json
import os

from Exp3_Config import Training_Config
from torch.utils.data import DataLoader
from Exp3_Model import TextCNN_Model
import torch
import numpy as np
from torchtext import data
from torchtext.vocab import build_vocab_from_iterator
from torch.nn import init
from tqdm import tqdm
from collections import Counter
import jieba
from Exp3_Config import Training_Config
import torch.nn as nn
from Exp3_Config import Training_Config
config = Training_Config()
class SentenceProcess:
    def __init__(self):
        pass


    def load_stopword(self):
        stop_words_file = open(config.path_root+'stop_words.txt', encoding='utf-8')
        stop_words = [line.strip() for line in stop_words_file]
        stop_words_file.close()
        return stop_words

    def tokenizer(self, sentence):
        assert os.path.exists(config.path_root+'jieba_vocab.txt'), 'there is no "jieba_vocab.txt" file'
        jieba.load_userdict(config.path_root+'jieba_vocab.txt')
        # 中文分词并且去停用词
        origin_words_list = jieba.cut(sentence)
        stop_words = self.load_stopword()
        words_list = []
        for word in origin_words_list:
            if word not in stop_words:
                if word !='\t':
                    words_list.append(word)
        return words_list  # list_sentence['你','我','他']

    def biuld_jieba_vocab(self,file_path):
        # jieba not good enough to split sentence
        # so we need to build a jieba_vocab to split our senttence
        '''

        :param file_path: train.txt
        :return: None
        '''
        lines = open(file_path, 'r', encoding='utf-8').readlines()
        for line in lines:
            line = line.split('\t')
            entity1 = line[0]
            entity2 = line[1]
            relation = line[2]
            with open(config.path_root+'jieba_vocab.txt', 'a', encoding='utf-8') as f:
                f.write('{}\n{}\n{}\n'.format(entity1, entity2, relation))

    def build_vocab(self, filepath, vocab_size):
        '''
        you need to run build_jieba_vocab() first, otherwise you will have no jieba_vocab.txt
        :param filepath: train.txt
        :param vocab_size: the max size of vocab
        :return: toechtext.vocab
        '''
        if not os.path.exists(config.path_root+'jieba_vocab.txt'):
            self.biuld_jieba_vocab(filepath)
        jieba.load_userdict(config.path_root+'jieba_vocab.txt')
        counter = Counter()
        with open(filepath,encoding='utf-8') as f:
            for line_ in f:
                counter.update(self.tokenizer(line_.strip()))
        counter_sorted = sorted(counter.items(),key=lambda x:x[1],reverse=True)
        counter_sorted = dict(counter_sorted[:vocab_size])
        return build_vocab_from_iterator([counter_sorted])

    def sentence2index(self,dictionary, sentence):
        # let look up in dictionary and convert words to number!
        sentence_id = []
        for word in sentence:
            if word in dictionary:
                sentence_id.append(dictionary[word])
            else:
                # sentence_id.append('[UNK]')
                a = np.random.randint(low=0, high=config.vocab_size)
                sentence_id.append(a)
        return sentence_id  # [1,2,3]

    def index_sentence_position_mask(self, original_data):  # original_data :Dict {head: tail: text: }
        '''

        :param original_data: Dict {head: tail: text: }
        :return:
        '''
        sentence = self.tokenizer(original_data['text'])
        # print('sentence',sentence)
        # print('len(sentence)',len(sentence))
        dictionary = np.load(config.path_root+'vocab.npy', allow_pickle=True).item()
        sentence_index = self.sentence2index(dictionary, sentence)
        # print('sentence_index',sentence_index)
        # print('len(sentence_index)',len(sentence_index))

        pos1, pos2 = [], []
        entity1 = original_data['head']
        entity2 = original_data['tail']

        # print('entity2',entity2)
        ent1pos = int(original_data['text'].index(entity1) / len(original_data['text']) * len(sentence))
        ent2pos = int(original_data['text'].index(entity2) / len(original_data['text']) * len(sentence))
        # print('ent1pos',ent1pos)
        # print('ent2pos',ent2pos)
        for idx, word in enumerate(sentence):
            position1 = self.get_position(idx - ent1pos)
            position2 = self.get_position(idx - ent2pos)

            pos1.append(position1)
            pos2.append(position2)
        #padding
        while len(pos1) < config.max_sentence_length:
            pos1.append(0)
        while len(pos2) < config.max_sentence_length:
            pos2.append(0)
        pos1 = pos1[:config.max_sentence_length]
        pos2 = pos2[:config.max_sentence_length]

        # Mask
        # we need mask to avoid padding part
        '''
        过短的句子可以通过 padding 增加到固定的长度，但是 padding 对
        应的字符只是为了统一长度，并没有实际的价值，因此希望在之后的计算中屏蔽它们，
        这时候就需要 Mask。
        '''
        mask = []
        pos_min = min(ent1pos, ent2pos)
        pos_max = max(ent1pos, ent2pos)
        for i in range(len(sentence_index)):
            if i <= pos_min:
                mask.append(1)
            elif i <= pos_max:
                mask.append(2)
            else:
                mask.append(3)
        # Padding
        while len(mask) < config.max_sentence_length:
            mask.append(0)
        mask = mask[:config.max_sentence_length]

        while len(sentence_index) < config.max_sentence_length:
            sentence_index.append(0)
        sentence_index = sentence_index[:config.max_sentence_length]

        # mask = torch.tensor(mask).long().unsqueeze(0)  # (1, L)
        pos1 = torch.tensor(pos1).long()#.unsqueeze(0)
        pos2 = torch.tensor(pos2).long()#.unsqueeze(0)
        # print(token)
        sentence_index = torch.tensor(sentence_index).long()#.unsqueeze(0)
        mask = torch.tensor(mask).long()#.unsqueeze(0)
        return [sentence_index, pos1, pos2, mask]

    def get_position(self,pos):
        # nn.embedding can be negative number
        '''
        : -limit ~ limit => 0 ~ limit * 2 + 2
        : <-20  => 0
        : -20 => 1
        : 20 => 41
        : >20 => 42
        :param pos:
        :return: positive number
        '''
        if pos < -config.pos_limit:
            return 0
        if - config.pos_limit <= pos <=config.pos_limit:
            return pos + config.pos_limit +1
        if pos > config.pos_limit:
            return config.pos_limit *2 + 1

class RelationAndId:
    def __init__(self):
        with open(config.path_root+'./data/rel2id.json', 'r', encoding='utf-8') as fp:
            json_data = json.load(fp)
        self.json_data = json_data

    def relation2id(self,relation):

        return self.json_data[1][relation]

    def id2relation(self,id):

        return self.json_data[0][id]


if __name__ == '__main__':
    print("数据预处理开始......")
    buildvocab = True
    # 过敏性紫癜
    if buildvocab:
        print("start build vocab........")
        st = SentenceProcess()
        vocab = st.build_vocab(config.path_root+'./data/data_train.txt',config.vocab_size)
        # print(vocab.get_stoi())
        dict = vocab.get_stoi()
        assert type(dict) != 'dict', 'not dict!'
        np.save('vocab.npy', dict)
        print('Successfully build vocab!\n Save as Dictionary in vocab.npy')

    print("数据预处理完毕！")
