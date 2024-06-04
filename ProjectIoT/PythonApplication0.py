# подключаем библиотеку компьютерного зрения 
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
#Условимся, что лицо не может двигаться по "наблюдаемой" "площади" 10х10 метров
#быстрее чем 300м/с, а само лицо условно покажется за 3 секунды на камере 90 раз
#(при фпс=30), если все 90 кадров найдут лицо (возможны обрывы и утрата лиц, это
#зависит от обученной нейросети). Так как для преодоления дистанции в 10метров лицо
#должно двигаться быстрее 300м/с, чтобы камера не смогла распознать между двумя
#соседними кадрами лицо, перемещающееся по данной "площади" и счетчик не смог бы в
#таком случае распознать пересекло ли лицо условную черту. Проще говоря: кадры не
#должны срываться, т.к. кадры, в которых нет лиц программа не обрабатывает

#Условимся, что людей(лица) СНИМАЕТ ОДНА КАМЕРА, А НЕ 2 КАК БЫЛО ПРЕДЛОЖЕНО (значит
#необходимо снимать(считать) людей, а не лица, т.к. по затылку мы не определим лицо,
#иначе, если все таки лицо снималось бы камерой и на выходе, то это значит в момент
#ухода из здания человек должен проходить через U-образный коридор, а не прямой, так,
#чтобы на месте поворота стояла камера(возможно широкоугольная), которая снимала бы
#лица, которые входят и выходят)

#контейнер для работы с лицами
faceBoxes={}
#для очистки массива

#если координата ниже черты в новом кадре, то чел прошёл вниз
def chel_proshel_vniz(Xcentroid, Ycentroid, frameWidth, frameHeight, schetchik, xxx, yyy):
    boolean1 = 0.7 * Xcentroid * frameHeight - Ycentroid * frameWidth + 0.15 * frameHeight * frameWidth <= 0
    boolean2 = 0.7 * xxx * frameHeight - yyy * frameWidth + 0.15 * frameHeight * frameWidth > 0
    #если точка была в предыдущем кадре выше прямой, а в текущем находится ниже прямой = прошла черту
    if boolean1 and boolean2:
        schetchik += 1
        print("Proshel vniz!!! ", schetchik)
    return schetchik

#если координата выше черты в новом кадре, то чел прошёл вверх
#def chel_proshel_vverh(Xcentroid, Ycentroid, frameWidth, frameHeight, schetchik):
#    if (Xcentroid * frameHeight * 0.15 - Xcentroid * frameHeight * 0.85 + Ycentroid * frameWidth - frameWidth * frameHeight * 0.15 < 0):
#        schetchik -= 1
#        print("Proshel vverh!!! ", schetchik)
#    return schetchik

#для поиска ближайшего
def find_correct(frameOpencvDnn, Xcentroid, Ycentroid, iteration_number = int(-1)):
    #только запись в глобальный массив точки
    if iteration_number > int(len(faceBoxes) - 1):
        faceBoxes[iteration_number] = (Xcentroid, Ycentroid)
        cv2.putText(frameOpencvDnn, str(iteration_number), (Xcentroid+5, Ycentroid), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return Xcentroid, Ycentroid
    else:
        #массив дистанций
        temp_d={}
        #находим соседей(из предыдущего кадра) у новоявленного(из текущего кадра) лица(или лица, которое переместилось)
        for i in range(len(faceBoxes)):
            x, y = faceBoxes[i]
            distance=float(scipy.spatial.distance.euclidean((x,y), (Xcentroid,Ycentroid)))
            temp_d[i]=(distance)
    
        #номер элемента с минимальной дистанцией
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

# функция определения лиц
#net            - модель определения лиц
#frame          - кадр, в котором нужно найти все лица
#conf_threshold - порог срабатывания распознавания(меньше если картинка плохая, больше если уверены что перед нами лицо)
#тестировал на камере Genius Widecam F100 (1080p, 30fps)
#0.7        - дальность 1.5  метра
#0.5        - дальность 1.65 метра
#0.2        - дальность 1.75 метра
#меньше 0.2 - дальность неизвестна, т.к. любой объект выделяет за лицо
def highlightFace(net, frame, iii, schetchik, conf_threshold=0.7):
    # делаем копию текущего кадра
    frameOpencvDnn=frame.copy()
    # высота и ширина кадра
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    # преобразуем картинку в двоичный пиксельный объект
    # при conf_threshold=0.2 и рамке 250х250 видит лицо с расстояния 1.5 метра
    #                   нагрузка на ЦП(intel i5 7700 4core) 50%+-4% от Python`а
    # при conf_threshold=0.2 и рамке 400х400 видит лицо с расстояния 1.95 метра (при рамке 300х300 видит 1.75)
    #                   нагрузка на ЦП(intel i5 7700 4core) 55%+-4% от Python`а
    # при conf_threshold=0.2 и рамке 700х700 видит лицо с расстояния 2+ метра(не измерял)
    #                   нагрузка на ЦП(intel i5 7700 4core) 68%+-4% от Python`а
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (400, 400), [104, 117, 123], True, False)
    # устанавливаем этот объект как входной параметр для нейросети
    net.setInput(blob)
    # выполняем прямой проход для распознавания лиц
    detections=net.forward()
    # Линия на кадре, которую пересекают точки(лица)
    cv2.line(frameOpencvDnn, (int(0), int(frameHeight * 0.15)), (int(frameWidth), int(frameHeight * 0.85)), (255, 0, 0), 1)
    # динамический массив переменных для рамок вокруг лица(либо точки лица)
    TempfaceBoxes={}
    Xcentroid=int(0)
    Ycentroid=int(0)
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
            Xcentroid=int((x1+x2)/2)
            Ycentroid=int((y1+y2)/2)
            # добавляем их в общую переменную
            TempfaceBoxes[i]=(Xcentroid,Ycentroid)
            # рисуем точку на кадре
            cv2.circle(frameOpencvDnn, (Xcentroid, Ycentroid), 1, (0,255,0), 2)

    if TempfaceBoxes:
        ssize = len(TempfaceBoxes)
        different = ssize - len(faceBoxes)
        for i in range(ssize):
            #если это не 1ый кадр, в котором нашли лицо, а 2 или больше, в котором также нашли лицо, то
            if iii > 0:
                Xcentroid, Ycentroid = TempfaceBoxes[i]
                #если лиц стало больше чем было, то новым лицам не нужно перезаписывать координаты
                if different > 0:
                    xxx, yyy = find_correct(frameOpencvDnn, Xcentroid, Ycentroid, i)
                    schetchik = chel_proshel_vniz(Xcentroid, Ycentroid, frameWidth, frameHeight, schetchik, xxx, yyy)
                #если лиц стало столько же или меньше, чем было
                else:
                    xxx, yyy = find_correct(frameOpencvDnn, Xcentroid, Ycentroid)
                    schetchik = chel_proshel_vniz(Xcentroid, Ycentroid, frameWidth, frameHeight, schetchik, xxx, yyy)
            #для 1 кадра
            else:
                faceBoxes[i]=TempfaceBoxes[i]
                cv2.putText(frameOpencvDnn, str(i), (Xcentroid+5, Ycentroid), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        #пока что не решил вопросы связанные с:
        #2.очищением ячеек массива faceBoxes
        #4. САМОЕ ГЛАВНОЕ: корректность распознавания множества людей(эта задача для распознавания 1 человека решена)

        iii += 1
    else:
        iii = 0
        #print("The frame is disrupted or there are no people")
    
    return frameOpencvDnn, TempfaceBoxes, iii, schetchik

# загружаем веса для распознавания лиц
faceProto="opencv_face_detector.pbtxt"
# и конфигурацию самой нейросети — слои и связи нейронов
faceModel="opencv_face_detector_uint8.pb"
#faceProto="MobileNetSSD_deploy.prototxt"
## и конфигурацию самой нейросети — слои и связи нейронов
#faceModel="MobileNetSSD_deploy.caffemodel"
# запускаем нейросеть по распознаванию лиц
faceNet=cv2.dnn.readNet(faceModel,faceProto)
# ОСНОВНОЙ ЦИКЛ ПРОГРАММЫ. Получаем видео с камеры
video=cv2.VideoCapture(0)
 #пока не нажата любая клавиша — выполняем цикл
 #если увеличивать параметр waitKey(х), то в значительной степени снижается нагрузка на ЦП 
 #х - количество миллисекунд дилея одного цикла. В данном примере если поставтить 33
 #(эквивалент для камеры 30fps), то нагрузка на ЦП падает с 55%+-4% до 41%+-2%
 #формула для расчета: х=1000/у, где у - количество  необходимых кадров в секунду

#для контейнера
iii=int(0)
#счетчик(отрицательные значения возможны, тут просто считает баланс вошедших-вышедших)
schetchik=int(0)
while cv2.waitKey(198)<0:
    #home()
    # получаем очередной кадр с камеры
    hasFrame,frame=video.read()
    # если кадра нет
    if not hasFrame:
        # останавливаемся и выходим из цикла
        cv2.waitKey()
        break
    
    # распознаём лица в кадре
    resultImg, TempfaceBoxes, iii, schetchik = highlightFace(faceNet, frame, iii, schetchik)
    
    #если лиц нет
    #if not TempfaceBoxes:
        # выводим в консоли, что лицо не найдено
        #print("I dont see any faces")
    # выводим картинку с камеры
    cv2.imshow("zxc", resultImg)