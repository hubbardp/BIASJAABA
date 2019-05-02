import json

Cropparams = {'ncells' : '10', 'npatches': '2', 'crop_flag': '1','interest_pnts': [[181, 166], [207, 85]]}


with open("Cropfrt_param.json", "w") as write_file:
    json.dump(Cropparams, write_file, indent=4)

