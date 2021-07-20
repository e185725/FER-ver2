
#動画から表情認識をする

import cv2
 
#動画を読込み
#カメラ等でストリーム再生の場合は引数に0等のデバイスIDを記述する
video = cv2.VideoCapture(0)
 
while video.isOpened():
    #フレームを読込み
    ret, frame = video.read()
 
    #フレームが読み込めなかった場合は終了（動画が終わると読み込めなくなる）
    if not ret: break
 
    #-----------------
    # 画像処理を記述する
    #-----------------
 
    #フレームの描画
    cv2.imshow('frame', frame)
 
    #qキーの押下で処理を中止
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
 
#メモリの解放
video.release()
cv2.destroyAllWindows()