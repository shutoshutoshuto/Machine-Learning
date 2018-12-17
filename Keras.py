import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
import os
from PIL import Image
from sklearn.model_selection import train_test_split



#ラーメンデータの読み込み
image_j = []
path = 'C:/Users/aizawa/Desktop/programing/2018 1101_zirou/hyouka_jiro/'
files = os.listdir(path)
for file in files[:]:
    if file != "Thumbs.db":
    	filepath = path+file
    	image = Image.open(filepath)
    	image = image.convert('L')
    	resize_width,resize_height = 100,100
    	image_j.append(np.array(image.resize((int(resize_width),int(resize_height)))))

image_jiro = np.array(image_j)
image_jiro_label = np.ones([image_jiro.shape[0]])


image_i = []
path = 'C:/Users/aizawa/Desktop/programing/2018 1101_zirou/hyouka_iekei/'
files = os.listdir(path)
for file in files[:]:
    if file != "Thumbs.db":
    	filepath = path+file
    	image = Image.open(filepath)
    	image = image.convert('L')
    	resize_width,resize_height = 100,100
    	image_i.append(np.array(image.resize((int(resize_width),int(resize_height)))))

image_iekei = np.array(image_i)
image_iekei_label = np.zeros([image_iekei.shape[0]])


images_jiro_iekei = np.vstack([image_jiro, image_iekei])
labels_jiro_iekei = np.hstack([image_jiro_label, image_iekei_label])
train_images, test_images, train_labels, test_labels = train_test_split(images_jiro_iekei,
                                                                        labels_jiro_iekei,
                                                                        test_size = 0.5,
                                                                        random_state=0)




#0~1にする
train_images = train_images / 255
test_images = test_images / 255

#ラベルに名前つけ
class_names = ['iekei','ziro']

#モデルを立てる
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(100,100)),
#    keras.layers.normalization.BatchNormalization(axis=-1,
#                                                  momentum=0.99, 
#                                                  epsilon=0.001, 
#                                                  center=True, 
#                                                  scale=True, 
#                                                  beta_initializer='ones', 
#                                                  moving_mean_initializer='zeros', 
#                                                  moving_variance_initializer='ones', 
#                                                  beta_regularizer=None, 
#                                                  gamma_regularizer=None, 
#                                                  beta_constraint=None, 
#                                                  gamma_constraint=None),

	keras.layers.Dense(20,activation = tensorflow.nn.relu),
	keras.layers.Dense(2,activation = tensorflow.nn.softmax)])

#モデルのコンパイル
model.compile(optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate = 0.01),
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

#学習実行
history = model.fit(train_images, train_labels, epochs=100, verbose=1 ,validation_data=(test_images, test_labels))

#テストデータで確認
test_loss , test_acc = model.evaluate(test_images,test_labels)
print(test_acc)




plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure(2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()







#first.tic()
