import torch
common_config = {
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'img_channel': 1,
    'img_width': 100,
    'img_height': 32,
    'num_classes': 66,  # 31 + 25 + 10 = 66
    'rnn_hidden': 256,
    'leaky_relu': False,
}
train_config = {
    'epochs': 50,
    'train_batch_size': 32,
    'val_batch_size': 64,
    'lr': 5e-5,
    'show_interval': 10,        # 打印训练结果的间隔
    'valid_interval': 500,      # 验证一次的间隔
    'save_interval': 3000,      # 保存模型的间隔
    'cpu_workers': 0,           # 调用加载数据的线程数量
    'reload_checkpoint': None,  # 预训练的模型
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': 'checkpoint/'
}
eval_config = {
    'eval_batch_size': 32,
    'cpu_workers': 0,
    'reload_checkpoint': None,
    'decode_method': 'beam_search',
    'beam_size': 10,
}
