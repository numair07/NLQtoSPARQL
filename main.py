from SPARQLWrapper import SPARQLWrapper, GET, JSON
from fuzzywuzzy import fuzz

sparql = SPARQLWrapper("http://localhost:7200/repositories/Priority_Companies")
sparql.setMethod(GET)
sparql.setReturnFormat(JSON)
query = "SELECT * WHERE { ?subject ?predicate ?object }"
sparql.setQuery(query)
ret = sparql.query()

data = ret.convert()

subjects = []
properties = []
objects = []

w3_string = "http://www.w3.org/"
proton_string = "http://proton.semanticweb.org/"

db_pedia_resource = "http://dbpedia.org/resource/"
db_pedia_property = "https://dbpedia.org/property/"

file_url = "file:/uploaded/generated/"

for dt in data["results"]["bindings"]:
    subject = dt['subject']['value']
    predicate = dt['predicate']['value']
    object = dt['object']['value']

    if w3_string in predicate:
        continue

    if w3_string not in subject and proton_string not in subject:
        subject = subject.replace(file_url, '')
        subjects.append(subject)

    if w3_string not in predicate and proton_string not in predicate:
        predicate = predicate.replace(db_pedia_property, '').replace(file_url, '')
        properties.append(predicate)

    if w3_string not in object and proton_string not in object:
        object = object.replace(db_pedia_resource, '').replace(file_url, '')
        objects.append(object)


#for i in range(len(subjects)):
#    print(subjects[i], properties[i], objects[i])

sub = set(subjects)
pro = set(properties)
obj = set(objects)

sub_check = input("Enter Company Name : ")
best_value = ""
best_score = 0

for i in sub:
    if "Stock" not in i and db_pedia_resource in i:
        if fuzz.partial_ratio(sub_check.lower(), i.lower()) > best_score:
            best_score = fuzz.partial_ratio(sub_check.lower(), i.lower())
            best_value = i

print(best_value)
print(best_score)
print("-----------------------------------------")
