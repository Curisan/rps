from __future__ import absolute_import    
from __future__ import division    
from __future__ import print_function    
    
import tensorflow as tf    
import  numpy as np    
import PIL.Image as Image       
import time    
import cv2    
import RPi.GPIO as GPIO    
import signal    
import atexit    
# 舵机旋转的角度    
def rotate(p, angle):    
    p.ChangeDutyCycle(7.5+10*angle/180)    
    time.sleep(1)    
    p.ChangeDutyCycle(0)    
# 石头剪刀布主程序    
def rps(model_dir, classes, p):    
        
    clicked = False      
    def onMouse(event, x, y, flags, param):      
        global clicked      
        if event == cv2.EVENT_LBUTTONUP:      
            clicked = True      
          
    cameraCapture = cv2.VideoCapture(0)    
    cameraCapture.set(3, 100) # 帧宽度    
    cameraCapture.set(4, 100) # 帧高度    
    cv2.namedWindow('MyWindow')      
    cv2.setMouseCallback('MyWindow', onMouse)      
          
    print('showing camera feed. Click window or press and key to stop.')      
    success, frame = cameraCapture.read()      
    print(success)      
    count = 0    
    flag = 0    
    
    saver = tf.train.import_meta_graph(model_dir+".meta")    
    with tf.Session() as sess:    
        saver.restore(sess, model_dir)    
        x = tf.get_default_graph().get_tensor_by_name("images:0")    
        keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")    
        y = tf.get_default_graph().get_tensor_by_name("fc2/output:0")    
        count=0    
        while success and cv2.waitKey(1)==-1 and not clicked:    
            time1 = time.time()    
            cv2.imshow('MyWindow', frame)      
            success, frame = cameraCapture.read()    
            img = Image.fromarray(frame)    
            # 将图片转化成灰度并缩小尺寸    
            img = np.array(img.convert('L').resize((28, 28)),dtype=np.float32)    
                
            img = img.reshape((1,28*28))    
            img = img/255.0 # 图像前处理    
            prediction = sess.run(y, feed_dict={x:img,keep_prob: 1.0})    
                
            index = np.argmax(prediction)    
            probability = prediction[0][index]    
            # 设置probability为0.8是为了提高识别稳定性    
            if index==1 and flag!=1 and probability>0.8:    
                  rotate(p, 30)    
                  flag=1    
            elif index==2 and flag!=2 and probability>0.8:     
                  rotate(p, 0)    
                  flag = 2    
            elif index==3 and flag!=3 and probability>0.8:    
                  rotate(p, -30)    
                  flag = 3    
            time2 = time.time()    
            if count%30==0:    
              print(classes[index])    
                  
              print('using time: ', time2-time1)    
               
              print('probability: %.3g' % probability)    
            count+=1    
              
        cv2.destroyWindow('MyWindow')      
        cameraCapture.release()     
    
    
if __name__=="__main__":    
  classes = ['others','paper','rock','scissors']    
  model_dir="model/model.ckpt"    
    
  atexit.register(GPIO.cleanup)        
          
  servopin = 21  # BCM模式下的21引脚  
    
  GPIO.setmode(GPIO.BCM)      
  GPIO.setup(servopin, GPIO.OUT, initial=False)      
  p = GPIO.PWM(servopin,50) #50HZ      
  p.start(0)      
  time.sleep(2)    
    
  rps(model_dir, classes, p)    
      
