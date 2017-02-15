import json
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from math import radians, cos, sin, asin, sqrt

data_file = open('/Users/anastasiapopova/Desktop/YANDEX/COMPANIES/mined')
data = [item for item in data_file.readlines()[:282500]]

def distance_km(a, b):
    lon1 = a[0][0]
    lat1 = a[0][1]
    lon2 = b[0][0]
    lat2 = b[0][1]
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km
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

    return current_row[n]
def compare_vectors(x, y, f):
    res = 1 if any([i in y for i in x]) else -1
    if f > 1:
        res *= -1
    return res

class Params():
    def __init__(self, a, b):
        self.equal_names_value_value = levenshtein_distance(a.names_value_value, b.names_value_value)
        self.equal_address_geo_id = compare_vectors(a.address_geo_id, b.address_geo_id, 3)
        self.equal_address_components_name_value= compare_vectors(a.address_components_name_value, b.address_components_name_value, 1)
        self.equal_address_formated_value= compare_vectors(a.address_formated_value, b.address_formated_value, 1)
        self.equal_address_pos_coordinates = distance_km(a.address_pos_coordinates, b.address_pos_coordinates)
        self.equal_duties = compare_vectors(a.duties, b.duties, 2)
        self.equal_email = compare_vectors(a.email, b.email, 2)
        self.equal_parent_companies = compare_vectors(a.parent_companies, b.parent_companies, 2)
        self.equal_phones_regioncode = compare_vectors(a.phones_regioncode, b.phones_regioncode, 3)
        self.equal_phones_number = compare_vectors(a.phones_number, b.phones_number, 2)
        self.equal_rubrics_id = compare_vectors(a.rubrics_id, b.rubrics_id, 2)
        self.equal_urls = compare_vectors(a.urls, b.urls, 2)
    
    
    def get_scores(self):
        return [self.equal_names_value_value, self.equal_address_geo_id, self.equal_address_components_name_value,
                self.equal_address_formated_value,  self.equal_address_pos_coordinates, self.equal_duties,
                self.equal_email, self.equal_parent_companies, self.equal_phones_regioncode,
                self.equal_phones_number, self.equal_rubrics_id, self.equal_urls]


class Company():
    def __init__(self, a):
        a = json.loads(a)
        
        self.names_value_value = []
        if 'names' in a:
            for j in range (0, len(a['names'])):
                if 'value' in a['names'][j]['value']:
                    self.names_value_value.append(a['names'][j]['value']['value'])
        
        self.address_geo_id = filter(id, [a.get('adress', {}).get('geo_id', None)])
        
        self.address_components_name_value = []
        if ('address' in a) and ('componentss' in a['address']):
            for j in range (0, len(a['address']['components'])):
                if 'name' in a['address']['components'][j] and 'value' in a['address']['components'][j]['name']:
                    self.address_components_name_value.append(a['address']['components'][j]['name']['value'])

        self.address_formated_value = filter(id, [a.get('address', {}).get('formatted' , {}).get('value', None)])
        self.address_pos_coordinates = filter(id, [a.get('address', {}).get('pos' , {}).get('coordinates', None)])
        self.duties = a.get('duties', [])
        self.email = a.get('emails', [])
        self.parent_companies = a.get('parent_companies', [])
        
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
    

        self.urls = a.get('urls', [])

    def comp(self, c):
        return(Params(self, c))

f = open('/Users/anastasiapopova/Desktop/YANDEX/COMPANIES/train_set_100.txt')
trainset = []
labels = []
for i in range(0, 100):
    trainset.append([f.readline(), f.readline()])
    labels.append(f.readline())
train_p = []
for item in trainset:
    a = Company(item[0])
    b = Company(item[1])
    v = a.comp(b)
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
    a = Company(item[0])
    b = Company(item[1])
    v = a.comp(b)
    test_p.append(v.get_scores())

clf = clf.fit(train_p, labels)
test_labels = clf.predict(test_p)

f = open('/Users/anastasiapopova/Desktop/YANDEX/COMPANIES/labels', 'w')
for l in test_labels:
    f.write(l)
f.close()
