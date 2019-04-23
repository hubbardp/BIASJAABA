import json

HOFparams = {} 
HOFparams = {'nbins': '8'}
HOFparams['lk'] = {'threshold': '0.000003', 'sigma' : {'smoothing' : '3.0', 'derivative' : '1.0'}}
HOFparams['cell'] = {'h' : '40', 'w' : '40'}


with open("HOFparam.json", "w") as write_file:
    json.dump(HOFparams, write_file, indent=4)

