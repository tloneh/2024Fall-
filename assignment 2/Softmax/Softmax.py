
def train_softmax(x_train, y_train, x_test, y_test, optimizer='sgd', epochs=10):
    """
    训练一个 Softmax 分类器。
    
    参数:
    - x_train: 训练数据
    - y_train: 训练标签
    - x_test: 测试数据
    - y_test: 测试标签
    - optimizer: 优化器（默认 'sgd'）
    - epochs: 训练轮次（默认 10）
    
    返回:
    - history: 模型训练历史
    """
    # 构建模型
    linear_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 编译模型
    linear_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    history = linear_model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    
    return history
