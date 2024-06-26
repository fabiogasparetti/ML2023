# -*- coding: utf-8 -*-
"""ML2023_00_colab_i/o.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ElVxi4DmyLeItv0Y_GrvEkQfcu0buDdz

File: ML2023_00_colab_i/o
Author: Fabio Gasparetti
Date: 2024-04-03

Description: Colab I/O - demo
"""

# Nel terminale attivare l'environment Anaconda e installare le librerie se occorre:
#
# https://www.anaconda.com/download

# conda create -n python3_11_7_uniroma3 python=3.11
# source activate python3_11_7_uniroma3

# conda install numpy
# conda install pandas
# conda install sqlalchemy

# Se non si impiega Anaconda:
#
# pip install sqlalchemy
# pip install pandas
# pip install numpy

import gspread

from oauth2client.client import GoogleCredentials
from google.colab import files

import urllib.request
import zipfile
from functools import partial
import os
import urllib

import pandas as pd

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String

"""Accesso aGoogle Drive"""

from google.colab import drive
drive.mount('/content/gdrive')

!touch "/content/gdrive/My Drive/sample_file.txt"

"""Accesso a Google Sheet"""

import gspread
from google.auth import default
creds, _ = default()

gc = gspread.authorize(creds)

wb = gc.create('demo')

ws = gc.open('demo').sheet1
ws

cells = ws.range('A1:D2')
cells

rows = ws.get_all_values()
rows

# per convertire i dati in un Pandas DataFrame:
# pd.DataFrame.from_records(rows)

"""Accesso a db SQL"""

engine = create_engine('sqlite:///college.db', echo = True)
meta = MetaData()

students = Table(
   'students', meta,
   Column('id', Integer, primary_key = True),
   Column('name', String),
   Column('lastname', String),
)
meta.create_all(engine)
print("these are columns in our table %s" %(students.columns.keys()))

ins = students.insert()
ins = students.insert().values(name = 'prudhvi', lastname = 'varma')
conn = engine.connect()
result = conn.execute(ins)
conn.commit()

conn.execute(students.insert(), [
   {'name':'Bhaskar', 'lastname' : 'guptha'},
   {'name':'vibhav','lastname' : 'kumar'},
   {'name':'prudhvi','lastname' : 'varma'},
   {'name':'manoj','lastname' : 'varma'},
])
conn.commit()

conn = engine.connect()
stmt = students.delete().where(students.c.name == 'manoj')
conn.execute(stmt)
s = students.select()
conn.execute(s)
conn.commit()

conn = engine.connect()
stmt=students.update().where(students.c.name=='vibhav').values(name='Dr.vaibhav')
conn.execute(stmt)
s = students.select()
conn.execute(s)
conn.commit()

s = students.select().where(students.c.name=="prudhvi")
result = conn.execute(s)
for row in result:
   print (row)
conn.commit()

s = students.select().where(students.c.id < 3)
result = conn.execute(s)
for row in result:
   print (row)
conn.commit()

"""Altre sorgenti a cui poter accedere:

* Google Cloud Storage (GCS)
* AWS S3
* Kaggle datasets

"""