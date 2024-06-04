from flask import Flask
from datetime import datetime

app = Flask(__name__)
#��� ��������� �� Flask, ������� ��������� �� �������� ����� ��������
@app.route("/")
#������� ��� ���������� ������ �� ����� � ����� �� �� ����
def home():
    #���� � �����, ���� ������������ �������
    with open('ProjectIoT/example.txt', 'r') as file:
        data = file.read()
        # ���������� �������� ������ �������
        # content ���������� ������
        return """
        <meta http-equiv="refresh" content="1" /> 
        """ + "Number of people included: " + data
        
# ������ ���������� �������
if __name__ == "__main__":
    app.run()
