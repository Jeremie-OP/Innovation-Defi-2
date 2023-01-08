import json
import xmltodict

with open("dev.xml", "rb") as data:
    dict_data = xmltodict.parse(data)
    json_data = json.dumps(dict_data, indent=2)
with open("dev.json", "w") as file:
    file.write(json_data)