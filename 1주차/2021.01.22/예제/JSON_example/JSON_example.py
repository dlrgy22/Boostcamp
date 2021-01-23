import json

with open("json_example.json", "r", encoding="utf8") as f:
    contents = f.read()
    json_data = json.loads(contents)

for element in json_data["employees"]:
    print(element["firstName"])