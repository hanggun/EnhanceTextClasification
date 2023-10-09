class thuc_config:
    pretrained_model_path = r'D:\PekingInfoResearch\pretrain_models\torch\chinese_roberta_wwm_ext'
    max_seq_len = 128
    model_save_dir = 'weights/best_model_thuc.pt'

    train_data_path = r'D:\open_data\classification\THUCNews\cnews.train.txt'
    dev_data_path = r'D:\open_data\classification\THUCNews\cnews.val.txt'
    test_data_path =r'D:\open_data\classification\THUCNews\cnews.test.txt'

    continue_train = False
    save_model = True
    aug = True

    valid_portion = 1
    batch_size = 8
    gradient_accumulation_steps = 1
    max_epoches = 3
    lr = 2e-5
    lr_end = 3e-7
    other_lr = 2e-4
    weight_decay = 0.002
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    valid_start_epoch = 2
    early_stop = 5


class agnews_config:
    pretrained_model_path = r'/home/zxa/ps/pretrain_models/bert_base_uncased'
    max_seq_len = 128
    model_save_dir = 'weights/best_model_ag.pt'

    train_data_path = r'/home/zxa/ps/open_data/classification/AGNews/train.csv'
    dev_data_path = r'/home/zxa/ps/open_data/classification/AGNews/dev.csv'
    test_data_path =r'/home/zxa/ps/open_data/classification/AGNews/test.csv'

    continue_train = False
    save_model = False
    aug = False

    valid_portion = 1
    batch_size = 16
    gradient_accumulation_steps = 1
    max_epoches = 5
    lr = 2e-5
    lr_end = 3e-7
    other_lr = 2e-4
    weight_decay = 0.002
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    valid_start_epoch = 0
    early_stop = 5


class imdb_config:
    pretrained_model_path = r'/home/zxa/ps/pretrain_models/roberta-base/'
    max_seq_len = 256
    model_save_dir = 'weights/best_model_imdb.pt'

    train_data_path = r'//home/zxa/ps/open_data/classification/aclImdb/train.txt'
    dev_data_path = r'/home/zxa/ps/open_data/classification/aclImdb/dev.txt'
    test_data_path =r'/home/zxa/ps/open_data/classification/aclImdb/test.txt'

    continue_train = False
    save_model = True

    lamb = 0.2 # 0.6-1都试过了，1最好
    valid_portion = 1
    batch_size = 8
    gradient_accumulation_steps = 4
    max_epoches = 10
    lr = 2e-5
    lr_end = 3e-7
    other_lr = 1e-3
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    valid_start_epoch = 0
    early_stop = 5


class ng20_config:
    pretrained_model_path = r'/home/zxa/ps/pretrain_models/roberta-base/'
    max_seq_len = 256
    model_save_dir = 'weights/best_model_20ng.pt'

    train_data_path = r'/home/zxa/ps/open_data/classification/20ng/train.txt'
    dev_data_path = r'/home/zxa/ps/open_data/classification/20ng/test.txt'
    test_data_path = r'/home/zxa/ps/open_data/classification/20ng/test.txt'

    continue_train = False
    save_model = True

    valid_portion = 1
    batch_size = 8
    gradient_accumulation_steps = 4
    max_epoches = 10
    lr = 2e-5
    lr_end = 3e-7
    other_lr = 1e-3
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    valid_start_epoch = 0
    early_stop = 5


class ohsumed_config:
    pretrained_model_path = r'/home/zxa/ps/pretrain_models/roberta-base/'
    max_seq_len = 192
    model_save_dir = 'weights/best_model_ohsumed.pt'

    train_data_path = r'/home/zxa/ps/open_data/classification/ohsumed/train.txt'
    dev_data_path = r'/home/zxa/ps/open_data/classification/ohsumed/test.txt'
    test_data_path = r'/home/zxa/ps/open_data/classification/ohsumed/test.txt'

    continue_train = False
    save_model = True
    rank = True

    lamb = 1 # 之前是0.2最好，现在是1最好，高一点点
    valid_portion = 1
    batch_size = 8
    gradient_accumulation_steps = 4
    max_epoches = 10
    lr = 2e-5
    lr_end = 3e-7
    other_lr = 1e-3
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    valid_start_epoch = 0
    early_stop = 5


class r8_config:
    pretrained_model_path = r'/home/zxa/ps/pretrain_models/roberta-base/'
    max_seq_len = 128
    model_save_dir = 'weights/best_model_r8_aug1_aug.pt'

    train_data_path = r'/home/zxa/ps/open_data/classification/R8/train.txt'
    dev_data_path = r'/home/zxa/ps/open_data/classification/R8/test.txt'
    test_data_path = r'/home/zxa/ps/open_data/classification/R8/test.txt'

    continue_train = True
    save_model = True
    rank = True

    lamb = 1
    valid_portion = 1
    batch_size = 8
    gradient_accumulation_steps = 4
    max_epoches = 10
    lr = 2e-5
    lr_end = 3e-7
    other_lr = 1e-3
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    valid_start_epoch = 0
    early_stop = 10


class r52_config:
    pretrained_model_path = r'/home/zxa/ps/pretrain_models/roberta-base/'
    max_seq_len = 128
    model_save_dir = 'weights/best_model_r52.pt'

    train_data_path = r'/home/zxa/ps/open_data/classification/R52/train.txt'
    dev_data_path = r'/home/zxa/ps/open_data/classification/R52/test.txt'
    test_data_path = r'/home/zxa/ps/open_data/classification/R52/test.txt'

    continue_train = False
    save_model = True
    rank = True

    lamb = 1
    valid_portion = 1
    batch_size = 8
    gradient_accumulation_steps = 4
    max_epoches = 10
    lr = 2e-5
    lr_end = 3e-7
    other_lr = 1e-3
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    valid_start_epoch = 0
    early_stop = 5


class mr_config:
    pretrained_model_path = r'/home/zxa/ps/pretrain_models/roberta-base/'
    max_seq_len = 32
    model_save_dir = 'weights/best_model_mr.pt'

    train_data_path = r'/home/zxa/ps/open_data/classification/mr/train.txt'
    dev_data_path = r'/home/zxa/ps/open_data/classification/mr/test.txt'
    test_data_path = r'/home/zxa/ps/open_data/classification/mr/test.txt'

    continue_train = False
    save_model = True
    rank = True

    lamb = 0.2 # 试了0.9和1，0.9最好
    valid_portion = 1
    batch_size = 8
    gradient_accumulation_steps = 4
    max_epoches = 10
    lr = 2e-5
    lr_end = 3e-7
    other_lr = 1e-3
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    valid_start_epoch = 0
    early_stop = 5