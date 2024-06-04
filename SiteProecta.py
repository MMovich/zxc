from flask import Flask
from datetime import datetime

app = Flask(__name__)
#это декоратор во Flask, который указывает на конечную точку маршрута
@app.route("/")
#функция для считывания данных из файла и вывод их на сайт
def home():
    #путь к файлу, куда записывается счетчик
    with open('ProjectIoT/example.txt', 'r') as file:
        data = file.read()
        # обновление страницы каждую секунду
        # content количество секунд
        return """
        <meta http-equiv="refresh" content="1" /> 
        """ + "Number of people included: " + data
        
# запуск локального сервера
if __name__ == "__main__":
    app.run()
