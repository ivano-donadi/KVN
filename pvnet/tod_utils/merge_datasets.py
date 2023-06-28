import json
import argparse
import os

parser = argparse.ArgumentParser(description='Tool for merging synthetic and real training sets',
                                     epilog="")

parser.add_argument('-r', '--real', 
                        help='Json file containing the real training data', required=True) 
parser.add_argument('-s', '--synthetic', 
                        help='Json file containing the synthetic training data', required=True) 
parser.add_argument('-o', '--output', 
                        help='Output dir', required=True) 

args = parser.parse_args()

with open(args.real, 'r') as real_fn:
    real_data = json.load(real_fn)

with open(args.synthetic, 'r') as syn_fn:
    syn_data = json.load(syn_fn)

real_imgs = real_data['images']
real_ann = real_data['annotations']
syn_imgs = syn_data['images']
syn_ann = syn_data['annotations']

merged_imgs = []
merged_imgs.extend(real_imgs)
merged_imgs.extend(syn_imgs)

merged_ann = []
merged_ann.extend(real_ann)
merged_ann.extend(syn_ann)


for i,obj in enumerate(zip(merged_imgs, merged_ann)):
    obj[0]['id'] = i
    obj[1]['id'] = i
    obj[1]['image_id'] = i

merged_json = {}
merged_json["images"] = merged_imgs
merged_json["annotations"] = merged_ann
merged_json["categories"] = real_data['categories']


with open(os.path.join(args.output,'merged.json'), 'w') as merged_fn:
    json.dump(merged_json,merged_fn)




