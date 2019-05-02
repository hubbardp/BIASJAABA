import json

Cropparams = {'ncells' : '10', 'npatches': '3', 'crop_flag': '1','interest_pnts': [[289, 104], [221, 42], [69,221]]}


with open("Cropsde_param.json", "w") as write_file:
    json.dump(Cropparams, write_file, indent=4)


