# ���������� ���������� ������������� ������ 
import cv2
import scipy.spatial.distance
#from flask import Flask

#app = Flask(__name__)

#counter = 0

#@app.route('/')
#def home():
#    global counter
#    counter += 1
#    return f'Counter: {counter}'

#if __name__ == '__main__':
#    app.run()

print(cv2.__version__)
#���������, ��� ���� �� ����� ��������� �� "�����������" "�������" 10�10 ������
#������� ��� 300�/�, � ���� ���� ������� ��������� �� 3 ������� �� ������ 90 ���
#(��� ���=30), ���� ��� 90 ������ ������ ���� (�������� ������ � ������ ���, ���
#������� �� ��������� ���������). ��� ��� ��� ����������� ��������� � 10������ ����
#������ ��������� ������� 300�/�, ����� ������ �� ������ ���������� ����� �����
#��������� ������� ����, �������������� �� ������ "�������" � ������� �� ���� �� �
#����� ������ ���������� ��������� �� ���� �������� �����. ����� ������: ����� ��
#������ ���������, �.�. �����, � ������� ��� ��� ��������� �� ������������

#���������, ��� �����(����) ������� ���� ������, � �� 2 ��� ���� ���������� (������
#���������� �������(�������) �����, � �� ����, �.�. �� ������� �� �� ��������� ����,
#�����, ���� ��� ���� ���� ��������� �� ������� � �� ������, �� ��� ������ � ������
#����� �� ������ ������� ������ ��������� ����� U-�������� �������, � �� ������, ���,
#����� �� ����� �������� ������ ������(�������� ��������������), ������� ������� ��
#����, ������� ������ � �������)

#��������� ��� ������ � ������
faceBoxes={}
#��� ������� �������

#���� ���������� ���� ����� � ����� �����, �� ��� ������ ����
def chel_proshel_vniz(Xcentroid, Ycentroid, frameWidth, frameHeight, schetchik, xxx, yyy):
    boolean1 = 0.7 * Xcentroid * frameHeight - Ycentroid * frameWidth + 0.15 * frameHeight * frameWidth <= 0
    boolean2 = 0.7 * xxx * frameHeight - yyy * frameWidth + 0.15 * frameHeight * frameWidth > 0
    #���� ����� ���� � ���������� ����� ���� ������, � � ������� ��������� ���� ������ = ������ �����
    if boolean1 and boolean2:
        schetchik += 1
        print("Proshel vniz!!! ", schetchik)
    return schetchik

#���� ���������� ���� ����� � ����� �����, �� ��� ������ �����
#def chel_proshel_vverh(Xcentroid, Ycentroid, frameWidth, frameHeight, schetchik):
#    if (Xcentroid * frameHeight * 0.15 - Xcentroid * frameHeight * 0.85 + Ycentroid * frameWidth - frameWidth * frameHeight * 0.15 < 0):
#        schetchik -= 1
#        print("Proshel vverh!!! ", schetchik)
#    return schetchik

#��� ������ ����������
def find_correct(frameOpencvDnn, Xcentroid, Ycentroid, iteration_number = int(-1)):
    #������ ������ � ���������� ������ �����
    if iteration_number > int(len(faceBoxes) - 1):
        faceBoxes[iteration_number] = (Xcentroid, Ycentroid)
        cv2.putText(frameOpencvDnn, str(iteration_number), (Xcentroid+5, Ycentroid), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return Xcentroid, Ycentroid
    else:
        #������ ���������
        temp_d={}
        #������� �������(�� ����������� �����) � �������������(�� �������� �����) ����(��� ����, ������� �������������)
        for i in range(len(faceBoxes)):
            x, y = faceBoxes[i]
            distance=float(scipy.spatial.distance.euclidean((x,y), (Xcentroid,Ycentroid)))
            temp_d[i]=(distance)
    
        #����� �������� � ����������� ����������
        min_d=int(0)
        for i in range(len(temp_d)):
            if temp_d[0] > temp_d[i]:
                temp_d[0] = temp_d[i]
                min_d=int(i)

        
        x, y = faceBoxes[min_d]
        #print("it was the same face as in the previous frame ", min_d)
        faceBoxes[min_d] = (Xcentroid, Ycentroid)
        cv2.putText(frameOpencvDnn, str(min_d), (Xcentroid+5, Ycentroid), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return x, y

# ������� ����������� ���
#net            - ������ ����������� ���
#frame          - ����, � ������� ����� ����� ��� ����
#conf_threshold - ����� ������������ �������������(������ ���� �������� ������, ������ ���� ������� ��� ����� ���� ����)
#���������� �� ������ Genius Widecam F100 (1080p, 30fps)
#0.7        - ��������� 1.5  �����
#0.5        - ��������� 1.65 �����
#0.2        - ��������� 1.75 �����
#������ 0.2 - ��������� ����������, �.�. ����� ������ �������� �� ����
def highlightFace(net, frame, iii, schetchik, conf_threshold=0.7):
    # ������ ����� �������� �����
    frameOpencvDnn=frame.copy()
    # ������ � ������ �����
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    # ����������� �������� � �������� ���������� ������
    # ��� conf_threshold=0.2 � ����� 250�250 ����� ���� � ���������� 1.5 �����
    #                   �������� �� ��(intel i5 7700 4core) 50%+-4% �� Python`�
    # ��� conf_threshold=0.2 � ����� 400�400 ����� ���� � ���������� 1.95 ����� (��� ����� 300�300 ����� 1.75)
    #                   �������� �� ��(intel i5 7700 4core) 55%+-4% �� Python`�
    # ��� conf_threshold=0.2 � ����� 700�700 ����� ���� � ���������� 2+ �����(�� �������)
    #                   �������� �� ��(intel i5 7700 4core) 68%+-4% �� Python`�
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (400, 400), [104, 117, 123], True, False)
    # ������������� ���� ������ ��� ������� �������� ��� ���������
    net.setInput(blob)
    # ��������� ������ ������ ��� ������������� ���
    detections=net.forward()
    # ����� �� �����, ������� ���������� �����(����)
    cv2.line(frameOpencvDnn, (int(0), int(frameHeight * 0.15)), (int(frameWidth), int(frameHeight * 0.85)), (255, 0, 0), 1)
    # ������������ ������ ���������� ��� ����� ������ ����(���� ����� ����)
    TempfaceBoxes={}
    Xcentroid=int(0)
    Ycentroid=int(0)
    # ���������� ��� ����� ����� �������������
    for i in range(detections.shape[2]):
        # �������� ��������� ���������� ��� ���������� ��������
        confidence=detections[0,0,i,2]
        # ���� ��������� ��������� ����� ������������ � ��� ����
        if confidence>conf_threshold:
            # ��������� ���������� �����
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            Xcentroid=int((x1+x2)/2)
            Ycentroid=int((y1+y2)/2)
            # ��������� �� � ����� ����������
            TempfaceBoxes[i]=(Xcentroid,Ycentroid)
            # ������ ����� �� �����
            cv2.circle(frameOpencvDnn, (Xcentroid, Ycentroid), 1, (0,255,0), 2)

    if TempfaceBoxes:
        ssize = len(TempfaceBoxes)
        different = ssize - len(faceBoxes)
        for i in range(ssize):
            #���� ��� �� 1�� ����, � ������� ����� ����, � 2 ��� ������, � ������� ����� ����� ����, ��
            if iii > 0:
                Xcentroid, Ycentroid = TempfaceBoxes[i]
                #���� ��� ����� ������ ��� ����, �� ����� ����� �� ����� �������������� ����������
                if different > 0:
                    xxx, yyy = find_correct(frameOpencvDnn, Xcentroid, Ycentroid, i)
                    schetchik = chel_proshel_vniz(Xcentroid, Ycentroid, frameWidth, frameHeight, schetchik, xxx, yyy)
                #���� ��� ����� ������� �� ��� ������, ��� ����
                else:
                    xxx, yyy = find_correct(frameOpencvDnn, Xcentroid, Ycentroid)
                    schetchik = chel_proshel_vniz(Xcentroid, Ycentroid, frameWidth, frameHeight, schetchik, xxx, yyy)
            #��� 1 �����
            else:
                faceBoxes[i]=TempfaceBoxes[i]
                cv2.putText(frameOpencvDnn, str(i), (Xcentroid+5, Ycentroid), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        #���� ��� �� ����� ������� ��������� �:
        #2.��������� ����� ������� faceBoxes
        #4. ����� �������: ������������ ������������� ��������� �����(��� ������ ��� ������������� 1 �������� ������)

        iii += 1
    else:
        iii = 0
        #print("The frame is disrupted or there are no people")
    
    return frameOpencvDnn, TempfaceBoxes, iii, schetchik

# ��������� ���� ��� ������������� ���
faceProto="opencv_face_detector.pbtxt"
# � ������������ ����� ��������� � ���� � ����� ��������
faceModel="opencv_face_detector_uint8.pb"
#faceProto="MobileNetSSD_deploy.prototxt"
## � ������������ ����� ��������� � ���� � ����� ��������
#faceModel="MobileNetSSD_deploy.caffemodel"
# ��������� ��������� �� ������������� ���
faceNet=cv2.dnn.readNet(faceModel,faceProto)
# �������� ���� ���������. �������� ����� � ������
video=cv2.VideoCapture(0)
 #���� �� ������ ����� ������� � ��������� ����
 #���� ����������� �������� waitKey(�), �� � ������������ ������� ��������� �������� �� �� 
 #� - ���������� ����������� ����� ������ �����. � ������ ������� ���� ���������� 33
 #(���������� ��� ������ 30fps), �� �������� �� �� ������ � 55%+-4% �� 41%+-2%
 #������� ��� �������: �=1000/�, ��� � - ����������  ����������� ������ � �������

#��� ����������
iii=int(0)
#�������(������������� �������� ��������, ��� ������ ������� ������ ��������-��������)
schetchik=int(0)
while cv2.waitKey(198)<0:
    #home()
    # �������� ��������� ���� � ������
    hasFrame,frame=video.read()
    # ���� ����� ���
    if not hasFrame:
        # ��������������� � ������� �� �����
        cv2.waitKey()
        break
    
    # ��������� ���� � �����
    resultImg, TempfaceBoxes, iii, schetchik = highlightFace(faceNet, frame, iii, schetchik)
    
    #���� ��� ���
    #if not TempfaceBoxes:
        # ������� � �������, ��� ���� �� �������
        #print("I dont see any faces")
    # ������� �������� � ������
    cv2.imshow("zxc", resultImg)