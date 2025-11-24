import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# --------------------------
# 1. 数据生成与预处理
# --------------------------
def create_simulated_data(data_dir, num_samples=1000):
    """
    创建模拟的驾驶数据（图像和转向角）
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)

    # 生成转向角（-45度到45度之间）
    steering_angles = np.random.uniform(-45, 45, num_samples)

    # 创建CSV文件
    data = {'image_path': [], 'steering_angle': []}

    for i in range(num_samples):
        # 生成模拟图像（道路场景）
        img = np.zeros((240, 320, 3), dtype=np.uint8)

        # 绘制道路
        cv2.rectangle(img, (100, 0), (220, 240), (100, 100, 100), -1)

        # 绘制车道线
        cv2.line(img, (130, 0), (160, 240), (255, 255, 255), 2)
        cv2.line(img, (190, 0), (160, 240), (255, 255, 255), 2)

        # 根据转向角调整车道线
        angle_rad = np.radians(steering_angles[i])
        offset = int(50 * np.tan(angle_rad))

        cv2.line(img, (130 + offset, 0), (160 + offset, 240), (0, 255, 0), 2)
        cv2.line(img, (190 + offset, 0), (160 + offset, 240), (0, 255, 0), 2)

        # 保存图像
        img_path = os.path.join(data_dir, 'images', f'{i:04d}.jpg')
        cv2.imwrite(img_path, img)

        data['image_path'].append(img_path)
        data['steering_angle'].append(steering_angles[i])

    # 保存CSV文件
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(data_dir, 'driving_log.csv'), index=False)
    print(f"生成 {num_samples} 个模拟样本")
    return df


# 创建模拟数据
data_dir = 'simulated_data'
df = create_simulated_data(data_dir, num_samples=2000)


# 数据增强函数
def augment_image(img, angle):
    """
    对图像进行增强处理
    """
    # 随机水平翻转
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
        angle = -angle

    # 随机调整亮度
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    brightness = np.random.uniform(0.7, 1.3)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 随机裁剪
    img = img[50:190, :, :]  # 裁剪天空和地面部分

    # 随机平移
    tx = np.random.randint(-20, 20)
    ty = np.random.randint(-10, 10)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return img, angle


# 数据生成器
def data_generator(df, batch_size=32, augment=True):
    """
    生成批量训练数据
    """
    while True:
        batch_images = []
        batch_angles = []

        # 随机打乱数据
        shuffled_df = df.sample(frac=1).reset_index(drop=True)

        for i in range(batch_size):
            # 获取图像路径和转向角
            img_path = shuffled_df.iloc[i]['image_path']
            angle = shuffled_df.iloc[i]['steering_angle']

            # 读取图像
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 数据增强
            if augment:
                img, angle = augment_image(img, angle)

            # 图像预处理
            img = cv2.resize(img, (200, 66))  # 适应NVIDIA模型输入
            img = img / 255.0  # 归一化

            batch_images.append(img)
            batch_angles.append(angle)

        yield np.array(batch_images), np.array(batch_angles)


# 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 创建数据生成器
train_generator = data_generator(train_df, batch_size=32, augment=True)
val_generator = data_generator(val_df, batch_size=32, augment=False)


# --------------------------
# 2. 构建深度学习模型
# --------------------------
def build_model():
    """
    构建基于CNN的转向角预测模型（参考NVIDIA架构）
    """
    model = Sequential()

    # 图像预处理层
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66, 200, 3)))

    # 卷积层
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # 全连接层
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))  # 输出转向角

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    return model


# 构建模型
model = build_model()
model.summary()

# --------------------------
# 3. 训练模型
# --------------------------
# 定义回调函数
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    verbose=1,
    restore_best_weights=True
)

# 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // 32,
    epochs=30,
    validation_data=val_generator,
    validation_steps=len(val_df) // 32,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# 绘制训练曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('MSE损失')
plt.legend()
plt.title('训练过程')
plt.show()

# --------------------------
# 4. 模型评估与实时预测
# --------------------------
# 加载最佳模型
best_model = load_model('best_model.h5')

# 在验证集上评估
val_loss = best_model.evaluate(val_generator, steps=len(val_df) // 32, verbose=1)
print(f"验证集损失: {val_loss:.4f}")


# 实时预测演示
def real_time_prediction(model):
    """
    使用摄像头进行实时转向角预测
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)  # 设置宽度
    cap.set(4, 240)  # 设置高度

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 图像预处理
        img = frame[50:190, :, :]  # 裁剪
        img = cv2.resize(img, (200, 66))  # 调整尺寸
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色空间
        img = img / 255.0  # 归一化
        img = np.expand_dims(img, axis=0)  # 增加批次维度

        # 预测转向角
        steering_angle = model.predict(img)[0][0]

        # 在图像上显示转向角
        cv2.putText(
            frame,
            f"Steering Angle: {steering_angle:.2f} deg",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # 显示图像
        cv2.imshow('Autonomous Driving', frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 运行实时预测
print("开始实时预测... 按'q'退出")
real_time_prediction(best_model)