# -*- coding: utf-8 -*-

##2018年10月4日更新

##ChromeDriverをpathを通した場所に保存する必要がある
##参考（http://zipsan.hatenablog.jp/entry/20180417/1523906068）
##chromeDriverのインストール場所
##参考（https://chromedriver.storage.googleapis.com/index.html?path=2.42/）
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import shutil
import time
import sys


t1 = time.time() 


##chromeの画面が出てくるのが邪魔なので消すためのおまじない
options = Options()
options.add_argument('--headless') 

#仮想ブラウザの立ち上げ
driver = webdriver.Chrome(chrome_options=options)

#PDCFAのログイン画面
loginUrl= "https://pdcfa.actiontc.jp/dict/member/main/login/lang=1"
driver.get(loginUrl)

#ここにPDCFAにログインするときに使うIDとPasswdを入れる
username = "aizawa"
password = "A-s-032369"
ID = 1

##----------ここから仮想ブラウザ上でログイン操作を行う
#IDを入力フォームに記入している
userNameField = driver.find_element_by_xpath('//*[@id="post_LoginId"]')#IDのX_path
userNameField.send_keys(username)

#passwordを入力フォームに記入している
passwordField = driver.find_element_by_xpath('//*[@id="post_Passwd"]')#PasswdのX_path
passwordField.send_keys(password)

#ここでログインボタンをクリック
submitButton = driver.find_element_by_class_name('submit')
submitButton.click()

##ここでログイン
#後々idの部分を変化させるようにして、全員分スクレイピングできるようにする
profile = "https://pdcfa.actiontc.jp/dict/member/dmemo/top"
driver.get(profile)
data = driver.page_source.encode('utf-8')
#ログインしたかどうかを確かめるためにログイン後の画面のスクショ
#driver.save_screenshot('screen.png')

#フォルダの作成と削除
path = 'C:/Users/aizawa/Desktop/programing/Scraping/'
folder = "PDCFA"
if os.path.exists(folder):
    shutil.rmtree(path + folder)
if not os.path.exists(folder):
    os.mkdir(folder)


cost1 = 0
cost2 = 0
cost3 = 0
cost98 = 99
cost99 = 99
cost100 = 99
for i in range(1,101):
    t3 = time.time() 
    #ここでログイン
    driver.get("https://pdcfa.actiontc.jp/dict/member/dmemo/top/id=" + str(i))
    data = driver.page_source.encode('utf-8')

    ##----------ここからスクレイピング開始
    pdcfa = BeautifulSoup(data, "lxml")
    #ここを回してIDから人を分けれるようにする
    if i == ID:
        name_ = pdcfa.find(class_ = "name").string
    else :
        name = pdcfa.find_all("a",href = "/dict/member/dmemo/top/id=" + str(i), limit = 7)
        name_ = name[6].string


    
    IDname = "ID_" + str(i) + "_" + name_

    #多分一言メモが何個あるかを数えている（ようになってるはず）
    memolength = pdcfa.find_all(class_ = "memoCont")
    emojilength = pdcfa.find_all(class_ = "feelIconImg")
    

    memo_byte = 0
    cost = 0
    ##一言の抽出
    for n in range(1, len(memolength)+1):#一言メモの個数だけループを回す
        #ここで一言メモの抽出を行っている。HTML内部のspanのid = umeom_text_...の部分が一言メモなので、そこを抜き出している
        No_ = "No" + str(n)
        
        memo = pdcfa.find("span",id = "umemo_text_"+str(n))
        #一言メモが消されていた場合、バグるのでそのための処理
        if memo == None:
            continue
        memo = memo.string
        if memo == None:
            continue
        #メモのバイト蓄積
        memo_byte = memo_byte + sys.getsizeof(memo)
        
        #スタンプの数カウント
        emoji = pdcfa.find("ol",id = "popStp_feelList_" + str(i) + "_" + str(n))
        if emoji == None:
            emoji_ = 0
        if not emoji == None:
            emoji_ = (len(emoji)-1) / 2
            
        #マックスコメント
        if cost < int(emoji_)/sys.getsizeof(memo):
            cost = int(emoji_)/sys.getsizeof(memo)
            maxmemo = memo
        
        
        #出力（コメント、スタンプ数、コスパ）
        filename = IDname + ".txt"
        with open(path + folder + "/" + filename,"a",encoding = "utf-8") as fileobj:
            fileobj.write(No_ + "\n" + memo + "\n" + 
                          "スタンプ数   " + str(int(emoji_)) + "\n" +
                          "コスパ       " + str(int(emoji_)/sys.getsizeof(memo)) + "\n\n")
    
    #出力（全スタンプ数、全バイト数、平均コスパ、マックスコスパコメント）
    with open(path + folder + "/" + filename,"a",encoding = "utf-8") as fileobj:
        fileobj.write("\n[全スタンプ数]\n" + str(len(emojilength)) + "\n\n" + 
                      "[合計バイト数]\n" + str(memo_byte) + "\n\n" +
                      "[コスパ]\n" + str(len(emojilength) / memo_byte) + "\n\n" +
                      "[マックスコスパコメント]\n" +str (maxmemo) + "\n" + str(cost) + "\n\n")
    
    
    #コスパランキング作成
    if cost1 < len(emojilength) / memo_byte:
        cost3 = cost2
        cost2 = cost1
        cost1 = len(emojilength) / memo_byte
        IDname1 = IDname
        
    if cost2 < len(emojilength) / memo_byte < cost1:
        cost3 = cost2
        cost2 = len(emojilength) / memo_byte
        IDname2 = IDname
        
    if cost3 < len(emojilength) / memo_byte < cost2:
        cost3 = len(emojilength) / memo_byte
        IDname3 = IDname
        
    if cost99 < len(emojilength) / memo_byte < cost98:
        cost98 = len(emojilength) / memo_byte
        IDname98 = IDname
    
    if cost100 < len(emojilength) / memo_byte < cost99:
        cost98 = cost99
        cost99 = len(emojilength) / memo_byte
        IDname99 = IDname
    
    if len(emojilength) / memo_byte < cost100:
        cost98 = cost99
        cost99 = cost100
        cost100 = len(emojilength) / memo_byte
        IDname100 = IDname
    
    
    t4 = time.time() 
    print("処理時間　　%dsec"%(t4-t3))
    
#出力（コスパランキング）
with open(path + folder + "/" + "コスパランキング.txt","w",encoding = "utf-8") as fileobj:
       fileobj.write("第1位\n" + str(IDname1) + "\n" + str(cost1) + "\n\n" + 
                     "第2位\n" + str(IDname2) + "\n" + str(cost2) + "\n\n" +
                     "第3位\n" + str(IDname3) + "\n" + str(cost3) + "\n\n" +
                     ":\n:\n:\n\n" +
                     "第98位\n" + str(IDname98) + "\n" + str(cost98) + "\n\n" + 
                     "第99位\n" + str(IDname99) + "\n" + str(cost99) + "\n\n" +
                     "第100位\n" + str(IDname100) + "\n" + str(cost100) + "\n\n")
    

t2 = time.time() 
ptime = t2 -t1
print("処理時間　　%dsec"%(ptime))