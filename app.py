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
    
    # 요청으로 받은 데이터 가져오기
    data = request.json

    # main 함수 호출을 위한 인자 생성
    args = {
        "input_dir": data["input_dir"],
        "im_path1": data["im_path1"],
        "im_path2": data["im_path2"],
        "output_dir": data["output_dir"],
        "warp_loss_with_prev_list": data["warp_loss_with_prev_list"],
        "save_all": data["save_all"],
        "version": data["version"],
        "flip_check": data["flip_check"]
    }

    # main 함수 실행
    main_program(args)
    
    print("end!!!!!!!!!! main.py")
    output = "Success"
    return jsonify(output=output)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
