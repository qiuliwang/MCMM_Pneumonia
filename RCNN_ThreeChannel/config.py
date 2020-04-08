'''
Created by Wang Qiuli, Li Zhihuan
2019/4/8

wangqiuli@cqu.edu.cn
'''


class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.cnn = 'resnet50'               # 'vgg16' or 'resnet50 or inception_v3'
        self.num_lstm_units = 128
        self.num_initalize_layers = 1    # 1 or 2
        self.dim_initalize_layer = 512
        self.num_attend_layers = 2       # 1 or 2
        self.dim_attend_layer = 512
        self.num_decode_layers = 2       # 1 or 2
        self.dim_decode_layer = 256

        self.phase = 'train'
        self.train_cnn = True

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3
        self.unet_drop_rate = 0.5
        self.attention_loss_factor = 0.01

        # about the optimization
        self.num_epochs = 15
        self.batch_size = 32
        self.optimizer = 'Momentum'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.initial_learning_rate = 0.0005
        
        self.learning_rate_decay_factor = 0.5
        self.num_steps_per_decay = 3000
        
        self.clip_gradients = 5.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6

        # about the saver
        self.save_period = 1000
        self.save_dir = './models/'
        self.summary_dir = './summary/'

        # about the rnn
        self.num_layers = 1
