
# coding: utf-8

# In[2]:

import json
import codecs
import requests
import pprint
import scipy
import pickle
import numpy as np
import pandas as pd
from __future__ import print_function
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from math import radians, cos, sin, asin, sqrt

import warnings
warnings.filterwarnings("ignore")


f = open('rubrics_similarity.csv', 'r')
rubrics_similarity = [i.replace(',','.').rstrip().split(';')[4:6]+i.replace(',','.').rstrip().split(';')[8:] for i in f.readlines()[1:]]
for i in rubrics_similarity:
    i[0] = int(i[0])
    i[1] = int(i[1])
    i[2] = float(i[2])
    i[3] = float(i[3])
f.close()


def distance_m(a, b):
    lon1 = a[0][0]
    lat1 = a[0][1]
    lon2 = b[0][0]
    lat2 = b[0][1]
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    m = 6367000 * c
    return m


def levenshtein_distance(a, b):
    a = a[0]
    b = b[0]
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current_row = range(n+1)
    for i in range(1, m+1):
        previous_row, current_row = current_row, [i]+[0]*n
        for j in range(1,n+1):
            add, delete, change = previous_row[j]+1, current_row[j-1]+1, previous_row[j-1]
            if a[j-1] != b[i-1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return float(current_row[n])


def compare_vectors(x, y, f):
    res = 1 if any([i in y for i in x]) else -1
    if f > 1:
          res *= -1
    return res



def compare_rubrics(a, b):
    #rez = 1e-8
    c = set(a).intersection(set(b))
    #print ((len(c)+1e-9)/(len(a)+len(b)))
    if len(a)+len(b)==0:
        return 0
    else:
        return (len(c)+1e-9)/(len(a)+len(b))


def get_rubrics_metrics(a, b):
    p_m = []
    c_m = []
    for i in a:
        for j in b:
            for line in rubrics_similarity:
                if (line[0]==i and line[1]==j) or (line[0]==j and line[1]==i):
                    c_m.append(line[2])
                    p_m.append(line[3])
    if len(p_m)==0:
        p_m.append(-1)
    if len(c_m)==0:
        c_m.append(-1)
    return np.mean(p_m), np.mean(c_m) 


class Params():
    def __init__(self, a, b):
        self.names_levenshtein_distance = levenshtein_distance(a.names_value_value, b.names_value_value)
        self.names_len_sum = len(a.names_value_value) + len(b.names_value_value)
        #self.names_hamming_distance = scipy.spatial.distance.hamming(a.names_value_value, b.names_value_value)
        self.equal_address_geo_id = compare_vectors(a.address_geo_id, b.address_geo_id, 3)
        self.equal_address_components_name_value= compare_vectors(a.address_components_name_value, b.address_components_name_value, 1)
        self.equal_address_formated_value= compare_vectors(a.address_formated_value, b.address_formated_value, 1)        
        self.equal_address_pos_coordinates = distance_m(a.address_pos_coordinates, b.address_pos_coordinates)       
        self.equal_duties = compare_vectors(a.duties, b.duties, 2)
        self.equal_email = compare_vectors(a.email, b.email, 2)
        self.equal_parent_companies = compare_vectors(a.parent_companies, b.parent_companies, 2)
        self.equal_phones_regioncode = compare_vectors(a.phones_regioncode, b.phones_regioncode, 3)
        self.equal_phones_number = compare_vectors(a.phones_number, b.phones_number, 2)
        self.equal_rubrics_id = compare_vectors(a.rubrics_id, b.rubrics_id, 2)
        self.rubrics_prob_metric, self.rubrics_corr_metric = get_rubrics_metrics(a.rubrics_id, b.rubrics_id)
        self.equal_rubrics = compare_rubrics(a.rubrics_id, b.rubrics_id)
        #self.rubrics_corr_metric = get_corr_metric(a.rubrics_id, b.rubrics_id)
        self.equal_urls = compare_vectors(a.urls, b.urls, 2)
        
        
    def get_scores(self):
        return [self.names_levenshtein_distance, self.names_len_sum, self.equal_address_geo_id, self.equal_address_components_name_value,
               self.equal_address_formated_value,  self.equal_address_pos_coordinates, self.equal_duties,
               self.equal_email, self.equal_parent_companies, self.equal_phones_regioncode,
               self.equal_phones_number, self.equal_rubrics_id, self.rubrics_prob_metric, self.rubrics_corr_metric, self.equal_rubrics, 
                self.equal_urls]


class Company():
    def __init__(self, a):                     
            self.names_value_value = filter(id, a.get('names.value.value', [])) 
            self.address_geo_id = filter(id, [a.get('adress.geo_id', None)]) 
            self.address_components_name_value = filter(id, a.get('address.components.name.value', [])) 
            self.address_formated_value = filter(id, a.get('address.formatted.value', None)) 
            self.address_pos_coordinates = filter(id, [a.get('address.pos.coordinates', None)])                 
            self.duties = a.get('duties', [])       
            self.email = a.get('emails', [])            
            self.parent_companies = a.get('parent_companies', []) 
            self.phones_regioncode = filter(id, a.get('phones.region_code', []))  
            self.phones_number = filter(id, a.get('phones.number', []))
            self.rubrics_id = filter(id, a.get('rubrics.rubric_id', []))
            self.urls = a.get('urls', [])
    
    def comp(self, c):
        return(Params(self, c))
    


filename = 'random_forest.sav'
clf = pickle.load(open(filename, 'rb'))


bound = 0.4  #граница
jsons = []
for item in open('mined').readlines()[:10]: #[:282500]:
    jsons.append(json.loads(item)) 


# In[17]:

names = []
for i in range (0, len(jsons)):
    name = jsons[i]['names'][0]['value']['value']
    rub = list()
    for j in range(0,len(jsons[i]['rubrics'])):
        rub.append(jsons[i]['rubrics'][j]['rubric_id'])
    if len(rub) > 0:
        #rows=500 - надо менять на большое, но не может компьютер нормально сам считать
        query = 'http://localhost:8983/solr/gettingstarted/select?indent=on&q="' + name + '"&&' + str(rub) +'"&wt=json'+ '&rows=500'
    else:
        query = 'http://localhost:8983/solr/gettingstarted/select?indent=on&q="' + name +'"&wt=json'+ '&rows=500'
    
    response = json.loads(requests.get(query).text)['response']['docs']

    f = open(str(i) + '_rez_test.json', 'w')
    for i in range(0, len(response)-1):    
        for j in range(i+1, len(response)):        
            a = Company(response[i])
            b = Company(response[j])
            v = a.comp(b)
            probability = clf.predict_proba(v.get_scores())[0][1]
            if probability > bound:
                f.write(json.dumps(response[i], ensure_ascii=True))
                f.write('\n')
                f.write(json.dumps(response[j], ensure_ascii=True))
                f.write('\n')            
    f.close()



