#!/usr/bin/env python
# coding: utf-8

# In[31]:


GAS_List_Coordinat = [['55.554771','37.924931'],['60.765833','28.808552'],
                      ['55.798510','37.534730'],['55.862126','37.466772'],
                      ['55.848817','36.805567'],['53.041525','158.637171']]

Construction_List_Coordinat = [['55.558834','37.815781'],['55.900693','37.478917'],
                               ['56.359825','37.542558'],['53.064992','158.619518'],
                               ['55.847763','37.636684']]

Banks_List_Coordinat = [['55.728849','37.620321'],['56.342179','37.523720'],
                        ['56.007639','37.484526'],['55.782977','37.640659'],
                        ['53.019530','158.647842'],['55.630446','37.658377'],
                        ['55.633323','37.650055'],['55.909247','37.590461']]


# In[32]:


## Установка библиотек если потребуется
#pip install seaborn
import pandas as pd
# import datetime as dt
import sqlite3
# import matplotlib.pyplot as plt
import seaborn as sns
from math import pi,sqrt,sin,cos,atan2


# In[33]:


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import sqlite3


# ## Подключаемся к источнику

# In[34]:


conn = sqlite3.connect('work_mode.db')
query = '''
SELECT client_id, time, latitude, longitude, altitude, speed, course, sat, name
FROM work_mode;
'''
df = pd.read_sql_query(query,conn)


# ### Предобработка

# ##### Посмотрим на полученные данные

# In[35]:


df.head(15)


# ##### Изучим типы данных

# In[36]:


df.info()


# ##### Посчитаем количество дубликатов

# In[37]:


df.duplicated().sum()


# ##### Удалим дубли и проверим еще раз

# In[38]:


df = df.drop_duplicates().reset_index(drop = True)
df.duplicated().sum()


# ##### Количество уникальных записей:

# In[39]:


df.isnull().count()


# ##### Преобразуем client_id, sat в тип (Int), time в (datetime), столбцы speed,altitude и course в тип (float)

# In[40]:


df['client_id'] = df['client_id'].astype('int')
df['time'] = df['time'].apply(pd.to_datetime)
df['altitude'] = df['altitude'].astype('float')
df['speed'] = df['speed'].astype('float') 
df['course'] = df['course'].astype('float')
df['sat'] = df['sat'].astype('int')

df['latitude'] = df['latitude'].astype('float')
df['longitude'] = df['longitude'].astype('float')


# In[41]:


df.info()


# ##### Построим график "Ящик с усами" для определения аномалий в данных по скоростям

# In[42]:


ax = sns.boxplot(x="client_id", y="speed",
                 data=df, palette="Set3")


# In[43]:


# Сделаем срез данных по скорости
df = df.query('speed <= 80') #speed limit


# In[44]:


ax = sns.boxplot(x="client_id", y="speed",
                 data=df, palette="Set3")


# ##### Посмотрим на предобработанные данные

# In[45]:


df_1 = df.reset_index(drop=True)

# df['speed'] = pd.to_numeric(df.speed)
# df['latitude'] = pd.to_numeric(df.latitude)
# df['longitude'] = pd.to_numeric(df.longitude)

df_1 = df_1.groupby(by = ['client_id','time']).agg({'speed':'mean', 'latitude':'mean', 'longitude':'mean' })
#df_1 = df
df = df.sort_values(by=['client_id', 'time'])
df_1['speed_mean'] = df_1.speed.rolling(window=5).mean()
df_1.shape


# In[46]:


X_2 = df_1.dropna() # очистка пустых значений


# In[47]:


from sklearn.mixture import GaussianMixture # подготовка к обучениею модели кдастеризации EM алгоритм

gm = GaussianMixture(n_components=4, 
                       max_iter=100,
                    init_params='kmeans',
                    random_state=42)
X_2 = X_2.reset_index()
X_2 = X_2.set_index('time')
X_2 =X_2.drop(columns=['client_id'])
#X_2


# In[48]:


from sklearn.metrics import silhouette_score
gm.fit(X_2) #обучение модели
y_pred = gm.predict(X_2) # предсказание


# показываем, сколько кластеров и значений в них предсказано, -1 считаем выбросом
unique, counts = np.unique(y_pred, return_counts=True)
dict(zip(unique, counts))


# In[49]:


silhouette_score(X=X_2, labels=y_pred) # качество силуэта кластеризации


# In[50]:


X_4 = df_1.dropna()# добавление client_id в таблицу   
X_4 = X_4.reset_index()
X_5 = X_4.client_id
X_2 = X_2.reset_index()
X_2['client_id'] = X_5


# In[51]:


df_preg = pd.Series(data=y_pred, name='prediction') # добавление предсказаний в таблицу    
X_3 = pd.DataFrame({'y_pred':df_preg})  
X_2 = X_2.reset_index()
X_2['predict'] = X_3


# In[52]:


plt.rcParams['figure.figsize'] = 10, 10
plt.scatter(X_2['speed'], X_2['client_id'], c=y_pred, alpha=0.5, label='class 2')
plt.show() # визуаливация кластеризации - разбивка клиент и его скорость (видно профиль передвижений), желтый - авто, зеленый - гонщик?, 
#синий - общественный транспорт, фиолетовый - пешеход


# In[53]:


df = X_2


# In[54]:


df.head(15)


# Добавим столбцы с координатами всех точек

# In[55]:


# Список POI
poi_list = []
poi_name = ''
i = 0
for poi in GAS_List_Coordinat:
    latitude = poi[0]
    longitude = poi[1]
    poi_name = f'GAS_{i}'
    df[f'{poi_name}_latitude'] = float(latitude)
    df[f'{poi_name}_longitude'] = float(longitude)
    poi_list.append(f'{poi_name}')
    i = i+1
    
poi_name = ''
i = 0
for poi in Construction_List_Coordinat:
    latitude = poi[0]
    longitude = poi[1]
    poi_name = f'Construction_{i}'
    df[f'{poi_name}_latitude'] = float(latitude)
    df[f'{poi_name}_longitude'] = float(longitude)
    poi_list.append(f'{poi_name}')
    i = i+1
    
poi_name = ''
i = 0
for poi in Banks_List_Coordinat:
    latitude = poi[0]
    longitude = poi[1]
    poi_name = f'Bank_{i}'
    df[f'{poi_name}_latitude'] = float(latitude)
    df[f'{poi_name}_longitude'] = float(longitude)
    poi_list.append(f'{poi_name}')
    i = i+1


# In[56]:


def get_dist(row):
    for poi in poi_list:
        lat1 = row[f'{poi}_latitude']
        long1 = row[f'{poi}_longitude']
        lat2 = row['latitude']
        long2 = row[f'longitude']

        degree_to_rad = float(pi / 180.0)

        d_lat = (lat2 - lat1) * degree_to_rad
        d_long = (long2 - long1) * degree_to_rad

        a = pow(sin(d_lat / 2), 2) + cos(lat1 * degree_to_rad) * cos(lat2 * degree_to_rad) * pow(sin(d_long / 2), 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        m = 6371000 * c
        
        df[f'm_to_{poi}'] = float(m)
    return m


# ## Расчитываем расстояние до POI

# In[57]:


df['t'] = df.apply(get_dist,axis=1)
df = df.drop('t',axis = True)


# In[58]:


df


# ## Вот тут логика которую реализовать не успели реализовать - должна быть запись в БД триггеров

# In[59]:


if poi == 'Заправка' m_to_poi < 100 and predict == 3 and speed < 20:
    write_to_sql('коиент едет на заправку')

if m_to_poi < 100 and speed < 20:
    write_to_sql('коиент едет на заправку')

