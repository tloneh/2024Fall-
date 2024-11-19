import tensorflow as tf

class MLPModel(tf.keras.Model):
    def __init__(self, hidden_layers, units, num_classes=10):
        """
        多层感知机模型类
        
        参数:
        - hidden_layers: 隐藏层数
        - units: 每层的神经元数量
        - num_classes: 输出类别数量（默认为10）
        """
        super(MLPModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(32, 32, 3))
        self.hidden_layers = [
            tf.keras.layers.Dense(units, activation='relu') for _ in range(hidden_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        """
        定义前向传播逻辑
        
        参数:
        - inputs: 输入张量
        - training: 是否为训练模式
        
        返回:
        - 输出张量
        """
        x = self.flatten(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

  
