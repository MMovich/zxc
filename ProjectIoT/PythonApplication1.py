# ���������� ���������� ������������� ������ 
import cv2
import scipy.spatial.distance
#from flask import Flask, render_template, Response
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
def del_faces_from_Box():
    print("YDALENIE")
    for i in range(len(faceBoxes)):
        del faceBoxes[i]
        print("Chel udalen iz mass ", i)

#���� ���������� ���� ����� � ����� �����, �� ��� ������ ����
def chel_proshel_vniz(Xcentroid, Ycentroid, frameWidth, frameHeight, schetchik):
    if (Xcentroid * frameHeight * 0.15 - Xcentroid * frameHeight * 0.85 + Ycentroid * frameWidth - frameWidth * frameHeight * 0.15 >= 0):
        schetchik += 1
        print("Proshel vniz!!! ", schetchik)
    #return schetchik

#���� ���������� ���� ����� � ����� �����, �� ��� ������ �����
def chel_proshel_vverh(Xcentroid, Ycentroid, frameWidth, frameHeight, schetchik):
    if (Xcentroid * frameHeight * 0.15 - Xcentroid * frameHeight * 0.85 + Ycentroid * frameWidth - frameWidth * frameHeight * 0.15 < 0):
        schetchik -= 1
        print("Proshel vverh!!! ", schetchik)
    return schetchik

#��� ������ ����������(������� ���������� �����)
def find_correct(Xcentroid,Ycentroid):
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

    #faceBoxes[min_d] = (Xcentroid, Ycentroid)
    print("it was the same face as in the previous frame ", min_d)
    return min_d

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
            #TempfaceBoxes.append([Xcentroid,Ycentroid])
            #���� ��� �� 1�� ����, � ������� ����� ���� � =>2 � ������� ����� ����� ����, ��
            #if iii > 0:
            #    min_rad = int(find_correct(Xcentroid,Ycentroid))
            #    chel_proshel_vniz(Xcentroid, Ycentroid, frameWidth, frameHeight, schetchik)
            #    #schetchik = chel_proshel_vverh(Xcentroid, Ycentroid, frameWidth, frameHeight, schetchik)
            #    faceBoxes[min_rad]=(Xcentroid,Ycentroid)
            #    cv2.putText(frameOpencvDnn, str(min_rad), (Xcentroid+5, Ycentroid), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            
            ##��� 1 ����� 
            #else:
            #    #��������� ��� ������ � ������
            #    faceBoxes[iii]=(Xcentroid,Ycentroid)
            #    cv2.putText(frameOpencvDnn, str(iii), (Xcentroid+5, Ycentroid), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            #    iii += 1
            # ������ ����� �� �����
            #cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/300)), 8)
            cv2.circle(frameOpencvDnn, (Xcentroid, Ycentroid), 1, (0,255,0), 2)
        #else:
        #    del_faces_from_Box()
    if TempfaceBoxes:
        for i in range(len(TempfaceBoxes)):
            #���� ��� �� 1�� ����, � ������� ����� ����, � =>2, � ������� ����� ����� ����, ��
            if iii > 0:
                Xcentroid, Ycentroid = TempfaceBoxes[i]
                min_rad = int(find_correct(Xcentroid, Ycentroid))
                faceBoxes[min_rad] = (Xcentroid, Ycentroid)
                chel_proshel_vniz(Xcentroid, Ycentroid, frameWidth, frameHeight, schetchik)
                #schetchik = chel_proshel_vverh(Xcentroid, Ycentroid, frameWidth, frameHeight, schetchik)
                cv2.putText(frameOpencvDnn, str(min_rad), (Xcentroid+5, Ycentroid), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            #��� 1 �����
            else:
                faceBoxes[i]=TempfaceBoxes[i]
                cv2.putText(frameOpencvDnn, str(i), (Xcentroid+5, Ycentroid), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        iii += 1
    
    return frameOpencvDnn, TempfaceBoxes, iii

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

    # �������� ��������� ���� � ������
    hasFrame,frame=video.read()
    # ���� ����� ���
    if not hasFrame:
        # ��������������� � ������� �� �����
        cv2.waitKey()
        break
    
    # ��������� ���� � �����
    resultImg, TempfaceBoxes, iii = highlightFace(faceNet, frame, iii, schetchik)
    
    #���� ��� ���
    if not TempfaceBoxes:
        # ������� � �������, ��� ���� �� �������
        print("I dont see any faces")
    # ������� �������� � ������
    cv2.imshow("zxc", resultImg)