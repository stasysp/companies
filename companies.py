import json

%matplotlib inline
from __future__ import print_function
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

data = []
for item in open('/Users/anastasiapopova/Desktop/YANDEX/COMPANIES/mined').readlines()[:282500]:
    data.append(json.loads(item))

def comp1(x, y, f):
    if (x) and (y):
        if f==1:
            c = 1
            for item in x:
                if item not in y:
                    c = -1
            return c
        if (f==2) or (f==3):
            c = -1
            for item in x:
                if item in y:
                    c = 1
            return c
    else: return 0

class Params():
    def __init__(self, a, b):
        self.equal_names_value_value = comp1(a.names_value_value, b.names_value_value, 1)
        self.equal_address_geo_id = comp1(a.address_geo_id, b.address_geo_id, 3)
        self.equal_address_components_name_value= comp1(a.address_components_name_value, b.address_components_name_value, 1)
        self.equal_address_formated_value= comp1(a.address_formated_value, b.address_formated_value, 1)
        self.equal_address_pos_coordinates = comp1(a.address_pos_coordinates, b.address_pos_coordinates, 3)
        self.equal_duties = comp1(a.duties, b.duties, 2)
        self.equal_email = comp1(a.email, b.email, 2)
        self.equal_owners = comp1(a.owners, b.owners, 2)
        self.equal_parent_companies = comp1(a.parent_companies, b.parent_companies, 2)
        self.equal_phones_regioncode = comp1(a.phones_regioncode, b.phones_regioncode, 3)
        self.equal_phones_number = comp1(a.phones_number, b.phones_number, 2)
        self.equal_rubrics_id = comp1(a.rubrics_id, b.rubrics_id, 2)
        self.equal_urls = comp1(a.urls, b.urls, 2)
    
    
    def get_scores(self):
        return [self.equal_names_value_value, self.equal_address_geo_id, self.equal_address_components_name_value,
                self.equal_address_formated_value,  self.equal_address_pos_coordinates, self.equal_duties,
                self.equal_email, self.equal_owners, self.equal_parent_companies, self.equal_phones_regioncode,
                self.equal_phones_number, self.equal_rubrics_id, self.equal_urls]

class Company():
    def __init__(self, a):
        self.names_value_value = []
        if 'names' in a:
            for j in range (0, len(a['names'])):
                if 'value' in a['names'][j]['value']:
                    self.names_value_value.append(a['names'][j]['value']['value'])
    
        self.address_geo_id = []
        if ('address' in a) and ('geo_id' in a['address']) and (a['address']['geo_id']):
            self.address_geo_id.append([a['address']['geo_id']])
        
        self.address_components_name_value = []
        if ('address' in a) and ('componentss' in a['address']):
            for j in range (0, len(a['address']['components'])):
                if 'name' in a['address']['components'][j] and 'value' in a['address']['components'][j]['name']:
                    self.address_components_name_value.append(a['address']['components'][j]['name']['value'])
        
        self.address_formated_value = []
        if ('address' in a) and ('formatted' in a['address']) and ('value' in a['address']['formatted']) and (a['address']['formatted']['value']):
            self.address_formated_value.append([a['address']['formatted']['value']])

self.address_pos_coordinates = []
    if ('address' in a) and ('pos' in a['address']) and ('coordinates' in a['address']['pos']):
        self.address_pos_coordinates.append([a['address']['pos']['coordinates']])
        
        self.duties = []
        if ('duties' in a) and (a['duties']):
            self.duties.append([a['duties']])
        
        self.email = []
        if ('emails' in a) and (a['emails']):
            self.email.append([a['emails']])
        
        self.owners = []
        if ('owners' in a) and (a['owners']):
            self.owners.append([a['owners']])
        
        self.parent_companies = []
        if ('parent_companies' in a) and (a['parent_companies']):
            self.parent_companies.append([a['parent_companies']])

        self.phones_regioncode = []
        if 'phones' in a:
            for j in range (0, len(a['phones'])):
                if 'region_code' in a['phones'][j]:
                    self.phones_regioncode.append(a['phones'][j]['region_code'])

        self.phones_number = []
        if 'phones' in a:
            for j in range (0, len(a['phones'])):
                if 'number' in a['phones'][j]:
                    self.phones_number.append(a['phones'][j]['number'])

self.rubrics_id = []
    if 'rubrics' in a:
        for j in range (0, len(a['rubrics'])):
            if 'rubric_id' in a['rubrics'][j]:
                self.rubrics_id.append((a['rubrics'][j]['rubric_id']))
        
        self.urls = []
        if ('urls' in a) and (a['urls']):
            self.urls.append([a['urls']])

def comp(self, c):
    b = Company(c)
        return(Params(self, b))


f = open('/Users/anastasiapopova/Desktop/YANDEX/COMPANIES/train_set_100.txt')
trainset = []
labels = []
for i in range(0, 100):
    trainset.append([f.readline(), f.readline()])
    labels.append(f.readline())
train_p = []
for item in trainset:
    a = Company(json.loads(item[0]))
    v = a.comp(json.loads(item[1]))
    train_p.append(v.get_scores())

clf = RandomForestClassifier(n_estimators=47, max_depth=None,
                             min_samples_split=37, random_state=0, min_weight_fraction_leaf=0.01)
scores = cross_val_score(clf, train_p, labels, cv = 20)
print(scores.mean())

f = open('/Users/anastasiapopova/Desktop/YANDEX/COMPANIES/test_set_22.txt')
testset = []
for i in range(0, 22):
    testset.append([f.readline(), f.readline()])

test_p = []
for item in testset:
    a = Company(json.loads(item[0]))
    v = a.comp(json.loads(item[1]))
    test_p.append(v.get_scores())

clf = clf.fit(train_p, labels)
test_labels = clf.predict(test_p)

f = open('/Users/anastasiapopova/Desktop/YANDEX/COMPANIES/labels', 'w')
for l in test_labels:
    f.write(l)
f.close()
