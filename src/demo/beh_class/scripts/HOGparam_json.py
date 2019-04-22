import json


HOGparam = {'nbins': '8'}
HOGparam['cell'] = {'h' : '40', 'w' : '40'}

with open("HOGparam.json", "w") as write_file:
    json.dump(HOGparam, write_file)
 
