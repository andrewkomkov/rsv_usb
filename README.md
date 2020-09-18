Инструменты для работы:
Python (https://www.python.org/downloads/)
Jupyter notebook (https://jupyter.org/install.html)
sqlite (Для установки открываем cmd, прописываем python и используем комманду pip install sqlite3 для установки
		либо в среде Jupyter прописать в строке pip install sqlite3 и выполнить ее)

Используемые библиотеки:
Для получения данных с Kafka, установите библиотеку: pip install kafka-python.
Далее импортируйте ее следующим образом:
from kafka import KafkaConsumer

import sqlite3
import json
import pandas as pd
import datetime as dt
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from math import pi,sqrt,sin,cos,atan2
import seaborn as sns

Запуск приложения:

Основная логика находится в файле Main.py - Предварительная обработка данных, кластеризация, расчет расстояния до POI, условия срабатываня бизнес триггеров, запись триггеров в БД.

Consumer.py - Создает локальную БД в sqlite и забирает сообщения\потоки из kafka.

Work_mode.db - файл БД sqlite, где хранятся сообщения\потоки из kafka.
