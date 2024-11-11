import torch
import cv2
import numpy 
import requests
from PIL import Image
from flask import Flask,render_template,Response
import time
import schedule
import shutil
import threading
import os
from playsound import playsound
from math import sin, cos, sqrt, atan2, radians
import pynmea2
import serial
model = torch.hub.load('ultralytics/yolov5', 'custom', path="evolve.pt") #load model 
model.classes = 0 #class 0 คือ person
app = Flask(__name__) #กำหนดชื่อflask
s=serial.Serial('COM5',9600,timeout=1) # รับค่าGPSจากUSBเสมือน(GPSของโทรศัพท์มือถือ)

print(os.getcwd())

#Function ที่กำหนดค่าก่อนส่งข้อความ
def line(): 
    print('\n-------start----------')

    #Function ที่ส่งข้อความ
    def linenoti(msg): 
        url = 'https://notify-api.line.me/api/notify' 
        token = '##### Token #########' ##### Token #########
        img = {'imageFile': open('./detec/save/rgb.jpg','rb')}
        data = {'message': msg} 
        headers = {'Authorization':'Bearer ' + token}
        session = requests.Session() 
        per = session.post(url, headers=headers, files=img, data=data) 
        print(per.text) 
    
    
    if (detecc.countper>0): 
            print('+++++++++++++do++++++++++')
            detecc.res.save(save_dir='./detec/save') 
            up=cv2.imread('./detec/save/image0.jpg') 
            Image.fromarray(up).save('./detec/save/rgb.jpg') 
            msg = 'พบคน '+str(detecc.countper)+' คน' 
            linenoti(msg)
            shutil.rmtree('./detec')
            print('+++++++++++++do++++++++++')
     
    print('============end=======')

#Function ที่อ่านภาพจากกล้องที่ทำงานร่วมกันกับการส่งข้อความ
def detecc(): 
    while (True): 
        
        _,frame=cap.read() 
        results=model(frame) 
        detecc.res=model(frame) 
        cy=results.pandas().xyxy[0] 
        detecc.countper=(cy.name == 'person').sum() 
        last=numpy.squeeze(results.render())
        cv2.putText(last,"person: "+str(detecc.countper), (0,40), 0, 1, (50,205,50),2)
        cv2.imshow('Person Detection',last ) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
    cap.release() 
    cv2.destroyAllWindows() 

#Function ที่ให้ทำงานทุกๆ 5 วิ เพื่อป้องกันไม่ให้ส่งข้อความเตือนถี่เกินไป
def sch1(): 
    schedule.every(5).seconds.do(line) 
    while True:
        schedule.run_pending() 
        time.sleep(1) 
        if (cap.isOpened()==False): 
            schedule.clear() 
            break 

#Function ที่ส่งเสียงเตือนบนอุปกรณ์
def sound():
    while (cap.isOpened()==True):
            if (detecc.countper>0): 
                print('\nsound is playing\n')
                playsound('./play.mp3',block=True)

#Function ที่อ่านค่า GPS จาก COM port    
def gps():
    while True:
        s.flushInput()
        s.flushOutput()
        p=s.readline()
        print(p.decode('utf-8'))
        try:
            x=p.decode('utf-8').strip()
            lll=pynmea2.parse(str(x))
            print(lll.latitude,lll.longitude)
            return lll.latitude,lll.longitude
        except:

            print("Error")
            continue

#Function ที่หาระยะทาง 
def fu():

    #Function ที่นำ GPS มาคำนวณหาระยะทางด้วยสูตร haversine
    def dis():
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        R = 6373.0
        distance = R * c * 1000
        return distance
    olat,olon = gps()
    lat1 = radians(olat)
    lon1 = radians(olon)
    time.sleep(5*60)
    nlat,nlon = gps()
    lat2 = radians(nlat)
    lon2 = radians(nlon)
    print(nlat,nlon)
    distance= dis()
    #link='https://www.google.com/maps/dir/'+str(olat)+','+str(olon)+'/'+ str(nlat)+','+ str(nlon)
    #webbrowser.open(link)
    print("Result:", distance,"meter")
    return distance

#Function ที่อ่านภาพจากกล้องและทำนายของ Flask
def frames():
    try:
        while True:  
            _,re=cap.read() 
            results=model(re) 
            frame=numpy.squeeze(results.render())
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame=cv2.imread('wait.jpg')
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except:
        frame=cv2.imread('wait.jpg')
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/live')
def video_feed():
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')

def web():
        app.run(port=5000,host='0.0.0.0',use_reloader=False)
ser=threading.Thread(target=web)
ser.daemon=True
ser.start()
first=False
i=1

while True:
    try:
        print("__________________________________start_____________________________________")
        distance = fu()
        cap = cv2.VideoCapture(2) 
        time.sleep(2)
        if (distance<=20 ):
            print("Running ",i)
            job1=threading.Thread(target=detecc) 
            job2=threading.Thread(target=sch1) 
            job3=threading.Thread(target=sound)
            job1.start()
            time.sleep(5)
            job2.start()
            job3.start()
            job1.join() 
            job2.join()  
            job3.join()
            i+=1
            break
    finally:
        print("___________________________________end______________________________________")
