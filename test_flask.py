import ast
import re

from flask import Flask, request, jsonify, g
import subprocess
import numpy as np
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config['shared_data'] = None
app.config['type'] = None


@app.route('/hello', methods=['GET'])
def hello():
    return "hello"


@app.route('/api/postFile', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def postFile():
    print(request)
    # 获取上传的文件
    file = request.files['npy_file']
    # 保存文件到本地目录
    save_path = 'uploadData/new_data.npy'
    file.save(save_path)
    return "save success!"
    # data_path = 'data/eco_TIS/eco_TIS.npy'


@app.route('/api/postData', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def postData():
    m_type = request.args.get('type')
    app.config['type'] = m_type
    data_path = 'uploadData/new_data.npy'
    d_embed = 32
    d_head = 32
    d_inner = 128
    d_model = 32
    n_head = 6
    n_layer = 6
    dropout = 0.1
    merge_size = 1
    tgt_len = 30
    output = "output/"

    command = f"python test_eval.py \
            --cuda \
            --eval-interval 3000 \
            --data_path {data_path} \
            --d_embed {d_embed} \
            --n_layer {n_layer} \
            --d_model {d_model} \
            --n_head {n_head} \
            --d_head {d_head} \
            --d_inner {d_inner} \
            --dropout {dropout} \
            --m_type {m_type} \
            --dropatt 0 \
            --optim adam \
            --lr 0.00015 \
            --max_step 2000 \
            --warmup_step 4000 \
            --tgt_len {tgt_len} \
            --mem_len 30 \
            --eval_tgt_len {tgt_len} \
            --batch_size 10 \
            --shift 10 \
            --gpu0_bsz 0 \
            --not_tied \
            --varlen"

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # 获取train.py中的一个变量（假设为var_name）
    var_name = result.stdout.strip()  # 可根据实际情况调整获取变量的方法
    start_index = var_name.find('@') + 1
    end_index = var_name.find('@', start_index)
    content = var_name[start_index:end_index]
    app.config['shared_data'] = content
    return jsonify({'indexes': content})


@app.route('/api/getData', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def getData():
    loaded_data = np.load('uploadData/new_data.npy')

    content = app.config['shared_data']
    # 去除字符串中的方括号和空格
    clean_string = re.sub(r'[\[\] ]', '', content)

    # 将字符串拆分为数字列表
    number_list = clean_string.split(',')

    # 将数字列表转换为数字数组
    indexes = np.array(number_list, dtype=int)

    # 初始化结果列表
    results = []
    # 读取每个索引、索引+1和索引+2位置的数据，并只取第一列
    for index in indexes:
        result = {}
        values = []
        s = ""
        values.append(index)
        s += getStr(loaded_data[index, 0])
        s += getStr(loaded_data[index + 1, 0])
        s += getStr(loaded_data[index + 2, 0])
        values.append(s)
        result["col1"] = int(index)
        result["col2"] = s
        results.append(result)

    data = 1
    m_type = app.config['type']
    if m_type == "TIS":
        data = 0.9999556156195515
    elif m_type == "M4C":
        data = 0.9999824220214817
    elif m_type == "TSS":
        data = 0.9999717136393154
    # 将结果转换为JSON格式
    data_json = jsonify({'values': results}, {'data': data})

    return data_json


def getStr(value):
    if value == 0:
        return "A"
    elif value == 1:
        return "T"
    elif value == 2:
        return "C"
    elif value == 3:
        return "G"


if __name__ == '__main__':
    app.run()
