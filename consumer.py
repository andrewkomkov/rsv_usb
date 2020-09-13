#!/usr/bin/env python
# coding: utf-8

# In[7]:


# pip install kafka-python
from kafka import KafkaConsumer
from kafka import KafkaProducer
from kafka import TopicPartition


# In[8]:


import sqlite3
import json


# In[9]:


conn = sqlite3.connect('work_mode.db')
cursor = conn.cursor()

# Создание таблицы
cursor.execute("""
            CREATE TABLE IF NOT EXISTS
            work_mode (
            "client_id" TEXT NOT NULL,
            time TEXT,
            latitude TEXT, longitude TEXT, altitude TEXT, speed TEXT, course TEXT, sat TEXT,name TEXT
            );
               """)
cursor.close()
conn.commit()
conn.close()


# In[18]:


conn = sqlite3.connect('work_mode.db')


# In[17]:


k = ''
consumer = KafkaConsumer(
    'input7',
    bootstrap_servers=['gpbtask.fun:9092'],
    api_version=(0, 10, 1),
    max_poll_records = 500000,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')))
for msg in consumer:
    
    k = msg.value
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO work_mode VALUES (?,?,?,?,?,?,?,?,?)", [k["client_id"], k["time"], k["latitude"],k["longitude"], k["altitude (m)"], k["speed (km/h)"],
                                                           k["course"], k["sat"], k["name"]])
    except:
        print('Значение пропущено')
        pass
    conn.commit()    
    cursor.close()


# In[15]:


conn.close()


# In[ ]:




