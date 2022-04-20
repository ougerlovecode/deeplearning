"""
该文件旨在配置训练过程中的各种参数
请按照自己的需求进行添加或者删除相应属性
"""


class Training_Config(object):
    def __init__(self,
                 word_dim=100,
                 vocab_size=20000,
                 training_epoch=200,
                 num_val=2,
                 max_sentence_length=40,
                 cuda=False,
                 label_num=44,
                 learning_rate=0.01,
                 batch_size=16,
                 dropout=0.5,
                 pos_limit=20,
                 pos_dim=1,
                 filter_num=230,
                 path_root='/content/drive/MyDrive/Colab Notebooks/Exp3/Exp3-Code/'):
        self.word_dim = word_dim  # 词向量的维度
        self.vocab_size = vocab_size  # 词汇表大小
        self.epoch = training_epoch  # 训练轮数
        self.num_val = num_val  # 经过几轮才开始验证
        self.max_sentence_length = max_sentence_length  # 句子最大长度
        self.label_num = label_num  # 分类标签个数
        self.lr = learning_rate  # 学习率
        self.batch_size = batch_size  # 批大小
        self.cuda = cuda  # 是否用CUDA
        self.dropout = dropout  # dropout概率
        self.pos_limit = pos_limit  # position range (-limit,limit)
        self.pos_dim = pos_dim  # position 的维度
        self.filter_num = filter_num
        self.path_root = path_root


