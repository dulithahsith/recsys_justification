import json

input_path = 'data/cloth/cloth_filter_flat_positived.large.json'
output_path = 'data/cloth/cloth_filter_flat_positive.large.json'

with open(input_path, 'r', encoding='utf8') as infile, open(output_path, 'w', encoding='utf8') as outfile:
    data = json.load(infile)  # This loads the whole array
    for obj in data:
        json.dump(obj, outfile)
        outfile.write('\n')