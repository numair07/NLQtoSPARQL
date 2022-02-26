from SPARQLWrapper import SPARQLWrapper, GET, JSON

sparql = SPARQLWrapper("http://Numair:7200/repositories/Priority_Companies_V2")
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
        subject = subject.replace(db_pedia_resource, '').replace(file_url, '')
        subjects.append(subject)

    if w3_string not in predicate and proton_string not in predicate:
        predicate = predicate.replace(db_pedia_property, '').replace(file_url, '')
        properties.append(predicate)

    if w3_string not in object and proton_string not in object:
        object = object.replace(db_pedia_resource, '').replace(file_url, '')
        objects.append(object)


for i in range(len(subjects)):
    print(subjects[i], properties[i], objects[i])
