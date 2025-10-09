import re
import json
path_txt = 'C:/Users/ro517/Рабочий стол/Univer/МАГ ИБ/mag_study/Библиотеки машинного обучения/Практика/temp/temp.txt'
path_ipynb = 'C:/Users/ro517/Рабочий стол/Univer/МАГ ИБ/mag_study/Библиотеки машинного обучения/Практика/temp/temp.ipynb'
def writeCodeBlock (text):
    f.write("""{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": """)
    f.write (json.dumps(text, ensure_ascii=False))
    f.write ("}, ")

with open(path_txt, 'r', errors='ignore', encoding='UTF-8') as f:
    nums = f.read().splitlines()

with open (path_ipynb, 'w', encoding='UTF-8') as f:
    f.write("""{
"metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "pygments_lexer": "ipython3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5,
 "cells": [
  """)

    buffer=[]
    for i in nums:
        if i[0] == " ": i = i[1:]
        if "" == i: i=""
        if i and (re.search(r"([a-zA-Z].*?['\"]).*?[А-Яа-яЁё].*?(['\"])", i) or re.search(r"#.*?[А-Яа-яЁё]", i) or not re.search(r"[А-Яа-яЁё]", i)):
            buffer.append(i+'\n')
        else:
            if buffer:
                writeCodeBlock(buffer)
            buffer = []

    if buffer:
        writeCodeBlock(buffer)

with open (path_ipynb, 'rb+') as f:
    f.seek(-2, 2)
    f.truncate()

with open (path_ipynb, 'a', encoding='UTF-8') as f:
    f.write("]}")

