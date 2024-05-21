# подключаем библиотеку компьютерного зрения 
import cv2

# функция определения лиц
#net            - модель определения лиц
#frame          - кадр, в котором нужно найти все лица
#conf_threshold - порог срабатывания распознавания(меньше если картинка плохая, больше если уверены что перед нами лицо)
#тестировал на камере Genius Widecam F100 (1080p, 30fps)
#0.7        - дальность 1.5  метра
#0.5        - дальность 1.65 метра
#0.2        - дальность 1.75 метра
#меньше 0.2 - дальность неизвестна, т.к. любой объект выделяет за лицо
def highlightFace(net, frame, conf_threshold=0.2):
    # делаем копию текущего кадра
    frameOpencvDnn=frame.copy()
    # высота и ширина кадра
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    # преобразуем картинку в двоичный пиксельный объект
    # при conf_threshold=0.2 и рамке 400х400 видит лицо с расстояния 1.95 метра (при рамке 300х300 видит 1.75)
    #                   нагрузка на ЦП(intel i5 7700 4core) 55%+-4% от Python`а
    # при conf_threshold=0.2 и рамке 250х250 видит лицо с расстояния 1.95 метра
    #                   нагрузка на ЦП(intel i5 7700 4core) 50%+-4% от Python`а
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (400, 400), [104, 117, 123], True, False)
    # устанавливаем этот объект как входной параметр для нейросети
    net.setInput(blob)
    # выполняем прямой проход для распознавания лиц
    detections=net.forward()
    # переменная для рамок вокруг лица
    faceBoxes=[]

    # перебираем все блоки после распознавания
    for i in range(detections.shape[2]):
        # получаем результат вычислений для очередного элемента
        confidence=detections[0,0,i,2]
        # если результат превышает порог срабатывания — это лицо
        if confidence>conf_threshold:
            # формируем координаты рамки
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            # добавляем их в общую переменную
            faceBoxes.append([x1,y1,x2,y2])
            # рисуем рамку на кадре
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    # возвращаем кадр с рамками
    return frameOpencvDnn,faceBoxes

# загружаем веса для распознавания лиц
faceProto="opencv_face_detector.pbtxt"
# и конфигурацию самой нейросети — слои и связи нейронов
faceModel="opencv_face_detector_uint8.pb"

# запускаем нейросеть по распознаванию лиц
faceNet=cv2.dnn.readNet(faceModel,faceProto)

# ОСНОВНОЙ ЦИКЛ ПРОГРАММЫ. Получаем видео с камеры
video=cv2.VideoCapture(0)
 #пока не нажата любая клавиша — выполняем цикл
while cv2.waitKey(1)<0:
    # получаем очередной кадр с камеры
    hasFrame,frame=video.read()
    # если кадра нет
    if not hasFrame:
        # останавливаемся и выходим из цикла
        cv2.waitKey()
        break

    # распознаём лица в кадре
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    # если лиц нет
    if not faceBoxes:
        # выводим в консоли, что лицо не найдено
        print("I dont see any faces")
    # выводим картинку с камеры``
    cv2.imshow("zxc", frame)