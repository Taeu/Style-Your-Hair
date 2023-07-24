from flask import Flask, jsonify, request
from main import main_program
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    output="Hello, Flask"
    return jsonify(output=output)


@app.route('/hair', methods=['POST'])
def handle_request():
    #main.py 실행코드를 직접 이부분에 추가
    #출력 결과를 얻은 후, JSON 형식으로 변환해 반환

    print("start!!! main.py")
    # main 함수 실행
    main_program()
    print("finish main.py!!!!!")
    output = "Success"
    return jsonify(output=output)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
