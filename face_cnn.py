
#畳み込みしてみる

###ライブラリなどの準備
import sys
sys.path.append('..')
import read_data
import glob 
import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers,models
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd 
from keras import backend as K

"""
I tensorflow/core/platform/cpu_feature_guard.cc:142] 
This TensorFlow binary is optimized with oneAPI Deep Neural Network Library 
(oneDNN) to use the following CPU instructions in performance-critical operations:  
AVX2 FMA
"""
#対処法
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

checkpoint_path = "Training_log/cp.ckpt"
checkpointdir = os.path.dirname(checkpoint_path)

# チェックポイントコールバックを作る
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)


#テストデータと訓練データを格納するための配列を用意する
train_data,train_label = read_data.read_data( read_data.train_name )
test_data ,test_label = read_data.read_data( read_data.test_file )

#データを学習できるように整形
train_data,test_data = train_data/255.0,test_data/255.0
train_data = train_data.reshape((28708,48,48,1))
test_data = test_data.reshape((3589,48,48,1))

###モデルの作成
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(48,48,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dropout(0.2))#後から追加
model.add(layers.Dense(7,activation="softmax"))

#モデルの可視化
plot_model(
    model,
    show_shapes=True,
    show_layer_names=True,
    to_file="model_cnn_dropout.png"
)

print(model.summary())
"""
#層の名前を入手しているだけ
>>>for i in model.layers:
    print(i.name)

conv2d
max_pooling2d
conv2d_1
max_pooling2d_1
conv2d_2
flatten
dense
dropout
dense_1
"""

###モデルをコンパイル
model.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
                )

#学習
#モデルで学習
epochs = 2
model.load_weights(checkpoint_path)
result = model.fit(train_data, train_label,
                    epochs=epochs,
                    validation_data=(test_data,test_label),
                    shuffle=True,
                    callbacks=[cp_callback])


# #正解率の可視化
# plt.plot(range(1, epochs+1), result.history['accuracy'], label="training")
# plt.plot(range(1, epochs+1), result.history['val_accuracy'], label="validation")
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

model.load_weights(checkpoint_path)
test_loss,test_acc = model.evaluate(test_data,test_label,verbose=2)
print(test_acc)

###正答率のグラフ化
predictions = model.predict(test_data[:5])
for i in range(5):
    #argmaxで二次元配列の列ごとの最大値を示すインデックスを返す
    #予測した値と実際の解
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    bar_label = [0,1,2,3,4,5,6]
    axs[0].imshow(test_data[i],"gray")
    axs[0].set_title(i)
    axs[1].bar(bar_label,predictions[i],color="orange",alpha = 0.7)
    axs[1].grid()
    print(predictions[i],test_label[i])
plt.show()


###ヒートマップの表示と保存

predictions = model.predict(test_data)
emotion = ["angry","disgust","fear","happy","sad","surprise","neutral"]
pred = [np.argmax(i) for i in predictions]
cm = confusion_matrix(test_label, pred)
test_len = np.array([[467],[56],[496],[895],[653],[415],[607]])
cm = cm / test_len
cm = np.round(cm,3)

cm = pd.DataFrame(data=cm, index=emotion, 
                           columns= emotion)

sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues',fmt="g")
plt.xlabel("Pre", fontsize=13)
plt.ylabel("True", fontsize=13)
plt.show()
plt.savefig('sklearn_confusion_matrix.png')


###フィルターの可視化
# 畳み込み層のみを抽出
conv_layers = [l.output for l in model.layers]
conv_model = models.Model(inputs=model.inputs, outputs=conv_layers)
# 畳み込み層の出力を取得
conv_outputs = conv_model.predict(test_data)

for i in range(len(conv_outputs)):
    print(f'layer {i}:{conv_outputs[i].shape}')

def plot_conv_outputs(outputs):
    filters = outputs.shape[2]#画像選択
    plt.figure(figsize=(7,7))
    for i in range(filters):
        plt.subplot(filters//8 + 1, 8, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f'filter {i}')
        plt.imshow(outputs[:,:,i])
    plt.show()
    plt.clf()
    plt.close()

#画像
n = -1
# 1層目 
plot_conv_outputs(conv_outputs[0][n])
# 2層目 
plot_conv_outputs(conv_outputs[1][n])
# 3層目 
plot_conv_outputs(conv_outputs[2][n])
# 4層目 
plot_conv_outputs(conv_outputs[3][n])
# 5層目 
plot_conv_outputs(conv_outputs[4][n])


###重みの可視化
def plot_conv_weights(filters):
    filter_num = filters.shape[3]
    try:
        for i in range(filter_num):
            plt.subplot(filter_num//6 + 1, 6, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f'filter {i}')
            plt.imshow(filters[:, :, 0, i])
        plt.show()
        plt.clf()
        plt.close()
    except:
        pass

# 1層目 (Conv2D)
plot_conv_weights(model.get_layer(name="conv2d").get_weights()[0])
# 2層目 
plot_conv_weights(model.get_layer(name="conv2d_1").get_weights()[0])
# 3層目 
plot_conv_weights(model.get_layer(name="conv2d_2").get_weights()[0])