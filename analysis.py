import json





results = json.load(open("results.json", "r"))

parameters = [0 for i in range(8)]

for result in results:
    if sum(result["custom_results"]) < sum(result["default_results"]):
        parameters[result["parameter_index"]]+=1

print(parameters)

