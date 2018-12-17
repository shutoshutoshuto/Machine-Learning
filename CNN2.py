import tensorflow as tf
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm 
import sys
import time


t1 = time.time() 

#トレーニング二郎の読み込み
resize_width,resize_height = 28, 28
image_trj = []
path = 'C:/Users/aizawa/Desktop/programing/2018 1101_zirou/jiro3/'
files = os.listdir(path)
for file in files[:]:
    if file != "Thumbs.db":
        filepath = path+file
        image = Image.open(filepath)
        image = image.convert('L')
        image_trj.append(np.array(image.resize((int(resize_width),int(resize_height)))).flatten().astype(np.float32)/255.0)

image_train_jiro = np.array(image_trj)
image_jiro_label1 = np.ones([len(image_train_jiro)])
image_jiro_label2 = np.zeros([len(image_train_jiro)])
image_jiro_label = np.c_[image_jiro_label1, image_jiro_label2]
image_train_jiro_label = np.ndarray.tolist(image_jiro_label)


#テスト二郎の読み込み
image_tej = []
path = 'C:/Users/aizawa/Desktop/programing/2018 1101_zirou/hyouka_jiro/'
files = os.listdir(path)
for file in files[:]:
    if file != "Thumbs.db":
    	filepath = path+file
    	image = Image.open(filepath)
    	image = image.convert('L')
    	image_tej.append(np.array(image.resize((int(resize_width),int(resize_height)))).flatten().astype(np.float32)/255.0)

image_test_jiro = np.array(image_tej)
image_jiro_label1 = np.ones([len(image_test_jiro)])
image_jiro_label2 = np.zeros([len(image_test_jiro)])
image_jiro_label = np.c_[image_jiro_label1, image_jiro_label2]
image_test_jiro_label = np.ndarray.tolist(image_jiro_label)





#トレーニング家系の読み込み
image_tri = []
image_iekei_label = []
path = 'C:/Users/aizawa/Desktop/programing/2018 1101_zirou/iekei2/'
files = os.listdir(path)
for file in files[:]:
    if file != "Thumbs.db":
        filepath = path+file
        image = Image.open(filepath)
        image = image.convert('L')
        image_tri.append(np.array(image.resize((int(resize_width),int(resize_height)))).flatten().astype(np.float32)/255.0)
       
image_train_iekei = np.array(image_tri)
image_iekei_label1 = np.zeros([len(image_train_iekei)])
image_iekei_label2 = np.ones([len(image_train_iekei)])
image_iekei_label = np.c_[image_iekei_label1, image_iekei_label2]
image_train_iekei_label = np.ndarray.tolist(image_iekei_label)


#テスト家系の読み込み
image_tei = []
image_iekei_label = []
path = 'C:/Users/aizawa/Desktop/programing/2018 1101_zirou/hyouka_iekei/'
files = os.listdir(path)
for file in files[:]:
    if file != "Thumbs.db":
        filepath = path+file
        image = Image.open(filepath)
        image = image.convert('L')
        image_tei.append(np.array(image.resize((int(resize_width),int(resize_height)))).flatten().astype(np.float32)/255.0)
       
image_test_iekei = np.array(image_tei)
image_iekei_label1 = np.zeros([len(image_test_iekei)])
image_iekei_label2 = np.ones([len(image_test_iekei)])
image_iekei_label = np.c_[image_iekei_label1, image_iekei_label2]
image_test_iekei_label = np.ndarray.tolist(image_iekei_label)



labels_num = len(image_jiro_label[0])



#テスト、トレーニングデータの並べ替え
images_train_jiro_iekei = np.vstack([image_train_jiro, image_train_iekei])
images_test_jiro_iekei = np.vstack([image_test_jiro, image_test_iekei])
images_train_jiro_iekei_label = np.vstack([image_train_jiro_label, image_train_iekei_label])
images_test_jiro_iekei_label = np.vstack([image_test_jiro_label, image_test_iekei_label])
images_train_label = np.c_[images_train_jiro_iekei_label, images_train_jiro_iekei]
images_test_label = np.c_[images_test_jiro_iekei_label, images_test_jiro_iekei]

idx_train = np.random.permutation(images_train_label.shape[0])
idx_test = np.random.permutation(images_test_label.shape[0])

i = 0
train = np.zeros([images_train_label.shape[0], images_train_label.shape[1]])
for idx in idx_train:
    train[i, :] = images_train_label[idx, :]
    i += 1

i = 0
test = np.zeros([images_test_label.shape[0], images_test_label.shape[1]])
for idx in idx_test:
    test[i, :] = images_test_label[idx, :]
    i += 1


train_images = train[:, 2:]
train_labels = train[:, :2]
test_images = test[:, 2:]
test_labels = test[:, :2]





#セッション
#実行するためのもの
sess = tf.InteractiveSession()

#プレスホルダー
#配列を格納する場所
x = tf.placeholder(tf.float32, shape=[None, resize_width*resize_height])
y_ = tf.placeholder(tf.float32, shape=[None, labels_num])


#重みの初期化
#±2σの切断正規分布からランダムに取り出したテンソルを生成する。
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#バイアスの初期化
#定数のテンサーを作成
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#畳み込み
#第1引数: input
#インプットデータを、4次元([batch, in_height, in_width, in_channels])のテンソルを渡す
#一番最初は画像を読み込んだ後、reshape関数で[-1, in_height, in_width, in_channels]と変換し、渡せばよい
#第2引数: filter
#畳込みでinputテンソルとの積和に使用するweightにあたる
#４次元[filter_height, filter_width, in_channels, channel_multiplier] のテンソルを渡す
#最後の引数のchannel_multiplierだけchannel数が拡張される
#第3引数: strides
#ストライド（=１画素ずつではなく、数画素ずつフィルタの適用範囲を計算するための値)を指定
#ただしstrides[0] = strides[3] = 1. とする必要があるため、指定は[1, stride, stride, 1]と先頭と最後は１固定とする
#第4引数: padding
#「'SAME'」か「'VALID'」を利用
#ゼロパディングを利用する場合はSAMEを指定
#  https://qiita.com/tadOne/items/b484ce9f973a9f80036e
#    ゼロパディング 
#    入力の特徴マップの周辺を０で埋めること
#    →なぜこのようなことをするかというと。普通に畳み込みを行うと端の領域は他の領域を比べて畳み込まれる回数が少なくなってしまうため。
#    メリット
#    端のデータに対する畳み込み回数が増えるので端の特徴も考慮されるようになる
#    畳み込み演算の回数が増えるのでパラメーターの更新が多く実行される
#    カーネルのサイズや、層の数を調整できる
#    → Convolution層とPooling層で出力サイズは次第に小さくなるのでゼロパディングでサイズを増やしたりすると層の数を増やすことができる。
#  https://qiita.com/knight0503/items/486b22d125841ac6307a
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#プーリング層
#第1引数: value
#inputデータ
#畳込み層からの出力データをそのまま与えれば良い
#第2引数: ksize
#プーリングサイズを指定
#3x3にしたい場合は[1, 3, 3, 1]とすればよい
#第３引数以降はconv2dと同じのため割愛
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def PutBar(per, barlen):
    perb = int(per/(100.0/barlen)*0.5)+1
    s = '\r'
    s += '|'
    s += '=' * perb
    s += '>'
    s += '-' * int((barlen*0.5 - perb))
    s += '|'
    s += ' ' + (str(per+1) + '%').rjust(4)
    
    sys.stdout.write(s)

    
    
#画像を格納する箱
def save_image(file_name, image_ndarray, cols=8):
    # 画像数, 幅, 高さ
    count, w, h = image_ndarray.shape
    # 縦に画像を配置する数
    rows = int((count - 1) / cols) + 1
    # 復数の画像を大きな画像に配置し直す
    canvas = Image.new("RGB", (w * cols + (cols - 1), h * rows + (rows - 1)), (0x80, 0x80, 0x80))
    for i, image in enumerate(image_ndarray):
        # 横の配置座標
        x_i = int(i % cols)
        x = int(x_i * w + x_i * 1)
        # 縦の配置座標
        y_i = int(i / cols)
        y = int(y_i * h + y_i * 1)
        out_image = Image.fromarray(np.uint8(image))
        canvas.paste(out_image, (x, y))
    canvas.save('images/' + file_name, "PNG")



def create_images(tag):
#    batch = mnist.train.next_batch(10)
    feed_dict = {x: batch_x, keep_prob: 1.0}

    # 畳み込み１層
    h_conv1_result = h_conv1.eval(feed_dict=feed_dict)
    for i, result in enumerate(h_conv1_result):
        images = channels_to_images(result)
        save_image("%02d_%s_h_conv1.png" % (i, tag), images)

    # プーリング１層
    h_pool1_result = h_pool1.eval(feed_dict=feed_dict)
    for i, result in enumerate(h_pool1_result):
        images = channels_to_images(result)
        save_image("%02d_%s_h_pool1.png" % (i, tag), images)

    # 畳み込み２層
    h_conv2_result = h_conv2.eval(feed_dict=feed_dict)
    for i, result in enumerate(h_conv2_result):
        images = channels_to_images(result)
        save_image("%02d_%s_h_conv2.png" % (i, tag), images)

    # プーリング２層
    h_pool2_result = h_pool2.eval(feed_dict=feed_dict)
    for i, result in enumerate(h_pool2_result):
        images = channels_to_images(result)
        save_image("%02d_%s_h_pool2.png" % (i, tag), images)

    print("Created images. tag =", tag)
    print("Number: ", [v.argmax() for v in batch_x])



def channels_to_images(channels):
    count = channels.shape[2]
    images = []
    for i in range(count):
        image = []
        for line in channels:
            out_line = [pix[i] for pix in line]
            image.append(out_line)
        images.append(image)
    return np.array(images) * 255






#プレイスホルダーのreshape
#第1引数: tensor
#inputとなるテンソルを指定
#第2引数: shape
#変えたい形を指定
#ただし-1を指定した場合には次元が削減されflattenとなる
#与えられた画像をNNで必要なheight, width, channelに変換する
#https://qiita.com/tadOne/items/b484ce9f973a9f80036e
x_image = tf.reshape(x, [-1,resize_width,resize_height,1])


#第一畳み込み層
#  5×5のフィルタ、チャネルの数、深さ32(32枚の画像)
layer1 =32
W_conv1 = weight_variable([5, 5, 1, layer1])
b_conv1 = bias_variable([layer1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)



#第二畳み込み層
layer2 = 64
W_conv2 = weight_variable([5, 5, layer1, layer2])
b_conv2 = bias_variable([layer2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)



#高密度結合層
#前の層の画像の高さ×横×深さを1024の1列の配列にしている
W_fc1 = weight_variable([int(resize_width/4 * resize_height/4 * layer2), 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, int(resize_width/4 * resize_height/4 * layer2)])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



#ドロップアウト
#学習に邪魔なノードを無視
#第1引数: x
#プーリング層からの出力（を正規化して？）をそのまま与えれば良い
#第2引数: keep_prob
#ドロップアウトする率
#公式ドキュメントには「The probability that each element is kept.」とあるから、残す率を指定すればよい（？）
#https://qiita.com/tadOne/items/b484ce9f973a9f80036e
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



#読み出し層
W_fc2 = weight_variable([1024, labels_num])
b_fc2 = bias_variable([labels_num])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



#モデルの訓練と評価

#交差エントロピー
#Deep Learningでは「交差エントロピー誤差」や「二乗誤差」を使用したりします。
#重要なのは、この後の処理である「誤差逆伝播」ができる関数であることのようです。
#https://qiita.com/mine820/items/f8a8c03ef1a7b390e372
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))


#勾配降下法の最適化
#GradientDescentOptimizer    勾配降下法によるオプティマイザー 
#AdagradOptimizer            AdaGrad法によるオプティマイザー 
#MomentumOptimizer           モメンタム法によるオプティマイザー 
#AdamOptimizer               Adam法 （これも有名ですね．） 
#FtrlOptimizer               （初めて耳にしました）"Follow the Regularized Leader" アルゴリズム 
#RMSPropOptimizer            （初めて耳にしました）学習率の調整を自動化したアルゴリズム 
#https://qiita.com/TomokIshii/items/f355d8e87d23ee8e0c7a
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


#予測の評価
#tf.argmax(y,1)は各インプットに対して最も確からしいラベルを返し、tf.argmax(y_,1)は正解のラベルを返します。
#そしてtf.equalで私たちの予測が当たっていたかを判定することができます。
#https://qiita.com/qooa/items/3719fec3cfe764674fb9
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))


#型の変換する
#tf.cast( 変換したいもの , 変換後の型 )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#実行
sess.run(tf.initialize_all_variables())
create_images("before")


#ミニバッチ学習
#バッチ学習　　学習データxがN個あるときに、N個のデータを全て用いて、それぞれのデータでの損失lの平均を計算し、それをデータ全体の損失Lとする学習
#確率的勾配法　　N個のデータx1,x2,...,xNからランダムに１つxiを選び出し、そのデータ１つに対する損失lをそのままLとする学習
#ミニバッチ学習　　全体を考慮したバッチ学習と、確率的勾配法の間を取ったのがミニバッチ学習であり、このとき学習データxがN個あるときに、ランダムなn(≤N)個のデータを使いLを求める学習
#             分類クラス数が多いほど、ミニバッチサイズを小さくすることが有効かも。一番主流。
#https://www.hellocybernetics.tech/entry/2017/07/08/152859
num_epoch = 1001
show = 100
num_data = train_images.shape[0]
batch_size = 16
Loss = []
Accuracy = []
for i in range(1,num_epoch):
    #num_dataをランダムに並び替え
    sff_idx = np.random.permutation(num_data)
    for idx in range(0, num_data, batch_size):
        #train_imagesにおいて idx から idx+batch_size までの配列番号のデータをbatchに格納
        batch_x = train_images[sff_idx[idx: idx + batch_size 
            if idx + batch_size < num_data else num_data]]
        batch_y = train_labels[sff_idx[idx: idx + batch_size
            if idx + batch_size < num_data else num_data]]
        
        if i%100!=0:
            PutBar(i%100, 100)
    
    #batch_x、batch_y を引数として cross_entropy を計算
    if i%3 == 0:
        train_loss = cross_entropy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
        train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
        
    #100ずつ表示
    if i%show == 0:
        print("\nstep %d,   training loss %g,   training accuracy %g\n"%(i, train_loss, train_accuracy)) 
        
    #batch_x、batch_y を引数として train_step に代入
    #https://seishin55.hatenablog.com/entry/2017/04/23/155707
    train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
    
    #LossとAccuracyの格納
    Loss.append(train_loss)
    Accuracy.append(train_accuracy)
#PutBar(100, 100)
create_images("after")

print("\ntest accuracy %g"%accuracy.eval(feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0}))
    
    

#　Lossの変化
plt.figure(1)
plt.plot(range(len(Loss)), Loss, 'b', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.legend()
plt.figure()
plt.show()

#　accuracyの変化
plt.figure(2)
plt.plot(range(len(Accuracy)), Accuracy, 'r', label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend()
plt.figure()
plt.show()


#各層におけるデータのサイズ
#feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0}
#print("W_conv1:      ", W_conv1.eval().shape)
#print("b_conv1:      ", b_conv1.eval().shape)
#print("x_image:      ", x_image.eval(feed_dict=feed_dict).shape)
#print("h_conv1:      ", h_conv1.eval(feed_dict=feed_dict).shape)
#print("h_pool1:      ", h_pool1.eval(feed_dict=feed_dict).shape)
#
#print("W_conv2:      ", W_conv2.eval().shape)
#print("b_conv2:      ", b_conv2.eval().shape)
#print("h_conv2:      ", h_conv2.eval(feed_dict=feed_dict).shape)
#print("h_pool2:      ", h_pool2.eval(feed_dict=feed_dict).shape)
#
#print("W_fc1:        ", W_fc1.eval().shape)
#print("b_fc1:        ", b_fc1.eval().shape)
#print("h_pool2_flat: ", h_pool2_flat.eval(feed_dict=feed_dict).shape)
#print("h_fc1:        ", h_fc1.eval(feed_dict=feed_dict).shape)
#
#print("h_fc1_drop: ", h_fc1_drop.eval(feed_dict=feed_dict).shape)
#
#print("W_fc2:      ", W_fc2.eval().shape)
#print("b_fc2:      ", b_fc2.eval().shape)
#print("y_conv:     ", y_conv.eval(feed_dict=feed_dict).shape)



images = x_image.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
images = images.reshape((-1, 28, 28)) * 255
save_image("!base.png", images)







t2 = time.time() 
ptime = t2 -t1
print("処理時間　　%dsec"%(ptime))