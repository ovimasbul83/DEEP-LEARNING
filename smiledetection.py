import cv2
face_csd = cv2.CascadeClassifier(r'C:\\Users\ovi\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
eye_csd = cv2.CascadeClassifier(r'C:\\Users\ovi\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_eye.xml')
smile_csd=cv2.CascadeClassifier(r'C:\\Users\ovi\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_smile.xml')
class smile:
    def smile_detect(self,gray,frame):
        face=face_csd.detectMultiScale(frame,1.09,10)
        for (x,y,w,h) in face:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            rgray=gray[y:y+h,x:x+w]
            rcolor=frame[y:y+h,x:x+w]
            smiled=smile_csd.detectMultiScale(rgray,1.8,19)
            for (sx,sy,sw,sh) in smiled:
                cv2.rectangle(rcolor, (sx, sy), (sx + sw, sy + sh), (250, 0, 0), 2)
        return frame
cap=cv2.VideoCapture(0)
while(True):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    m=smile()
    video=m.smile_detect(gray=gray,frame=frame)
    cv2.imshow('smile',video)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()