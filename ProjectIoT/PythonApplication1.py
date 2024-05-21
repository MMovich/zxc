# ���������� ���������� ������������� ������ 
import cv2

# ������� ����������� ���
#net            - ������ ����������� ���
#frame          - ����, � ������� ����� ����� ��� ����
#conf_threshold - ����� ������������ �������������(������ ���� �������� ������, ������ ���� ������� ��� ����� ���� ����)
#���������� �� ������ Genius Widecam F100 (1080p, 30fps)
#0.7        - ��������� 1.5  �����
#0.5        - ��������� 1.65 �����
#0.2        - ��������� 1.75 �����
#������ 0.2 - ��������� ����������, �.�. ����� ������ �������� �� ����
def highlightFace(net, frame, conf_threshold=0.2):
    # ������ ����� �������� �����
    frameOpencvDnn=frame.copy()
    # ������ � ������ �����
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    # ����������� �������� � �������� ���������� ������
    # ��� conf_threshold=0.2 � ����� 400�400 ����� ���� � ���������� 1.95 ����� (��� ����� 300�300 ����� 1.75)
    #                   �������� �� ��(intel i5 7700 4core) 55%+-4% �� Python`�
    # ��� conf_threshold=0.2 � ����� 250�250 ����� ���� � ���������� 1.95 �����
    #                   �������� �� ��(intel i5 7700 4core) 50%+-4% �� Python`�
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (400, 400), [104, 117, 123], True, False)
    # ������������� ���� ������ ��� ������� �������� ��� ���������
    net.setInput(blob)
    # ��������� ������ ������ ��� ������������� ���
    detections=net.forward()
    # ���������� ��� ����� ������ ����
    faceBoxes=[]

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
            # ��������� �� � ����� ����������
            faceBoxes.append([x1,y1,x2,y2])
            # ������ ����� �� �����
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    # ���������� ���� � �������
    return frameOpencvDnn,faceBoxes

# ��������� ���� ��� ������������� ���
faceProto="opencv_face_detector.pbtxt"
# � ������������ ����� ��������� � ���� � ����� ��������
faceModel="opencv_face_detector_uint8.pb"

# ��������� ��������� �� ������������� ���
faceNet=cv2.dnn.readNet(faceModel,faceProto)

# �������� ���� ���������. �������� ����� � ������
video=cv2.VideoCapture(0)
 #���� �� ������ ����� ������� � ��������� ����
while cv2.waitKey(1)<0:
    # �������� ��������� ���� � ������
    hasFrame,frame=video.read()
    # ���� ����� ���
    if not hasFrame:
        # ��������������� � ������� �� �����
        cv2.waitKey()
        break

    # ��������� ���� � �����
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    # ���� ��� ���
    if not faceBoxes:
        # ������� � �������, ��� ���� �� �������
        print("I dont see any faces")
    # ������� �������� � ������``
    cv2.imshow("zxc", frame)