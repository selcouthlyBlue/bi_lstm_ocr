class TrainConfig:
    def __init__(self):
        self.batch_size = 50
        self.log_dir = 'log/train'
        self.is_restore = False
        self.checkpoint_dir = 'ckpt/train'
        self.num_epochs = 5
