import msgpack, json, os

from PIL import Image
def gen_cmap(top, side):
	img = Image.new('RGBA', (1024, 1024), color=(0, 0, 0, 0))
	blank = Image.new('RGBA', (256, 256), color=(0, 0, 0, 0))

	# Ridiculous hack to get rid of weird white outlines
	if top:
		for i in range(0, img.width, top.width):
			for j in range(0, img.height, top.height):
				img.paste(top, (i, j))
	elif side:
		for i in range(0, side.width, side.width):
			for j in range(0, side.height, side.height):
				img.paste(top, (i, j))

	if top: img.paste(top, (384, 512))
	else: img.paste(blank, (384, 512))
	if side:
		img.paste(side, (384, 768))
		img.paste(side, (128, 768))
	else:
		img.paste(blank, (384, 768))
		img.paste(blank, (128, 768))
	
	return img

def transform(obj, keys):
    if isinstance(obj, bytes):
        return obj.decode('utf-8')
    
    if isinstance(obj, list):
        return [transform(element, keys) for element in obj]
    elif isinstance(obj, dict):
        return {keys[int(key)]: transform(value, keys) for key, value in obj.items()}
    else:
        return obj

def convert_msgpack(file):
	with open(file, 'rb') as in_file:
		return convert_msgpackfile(in_file)

def convert_msgpackfile(file):
	data = msgpack.unpack(file, strict_map_key=False)
		
	json_obj = data[0]
	keys = data[1]

	out = transform(json_obj, keys)
	
	return out

def correctDecimal(dec):
	hex_colour = hex(int(dec)).split('x')[-1]
	# Convert hex colour to RGB values
	# Colours encoded as BGR, not RGB
	b = (int(hex_colour, 16) & 0xFF0000) >> 16
	g = (int(hex_colour, 16) & 0x00FF00) >> 8
	r = int(hex_colour, 16) & 0x0000FF
	
	corrected = [r, g, b]

	return (int(corrected[0]) << 16) | (int(corrected[1]) << 8) | int(corrected[2])

def find(element, JSON, path, all_paths):    
  if element in JSON:
    path = path + element
    all_paths.append(path)
  for key in JSON:
    if isinstance(JSON[key], dict):
      find(element, JSON[key],path + key + '.',all_paths)

def get_locator(categories, boundless_path):
	with open(boundless_path + '/assets/archetypes/locatortemplates.json') as loc_templates, open(boundless_path + '/assets/archetypes/categoryhierarchy.json') as hierarchy:
		loc_templates_json = json.load(loc_templates)
		hierarchy_json = json.load(hierarchy)
	
	paths = []
	for category in categories:
		all_paths = []
		find(category,hierarchy_json,'',all_paths)
		paths += all_paths
	
	results = {}
	found = False
	for path, categories in loc_templates_json:
		f = 0
		bias = 0
		for c in categories:
			for p in paths:
				if c in p.split('.'):
					f += 1
					ind = p.split('.').index(c)
					if ind > bias: bias = ind
		if f >= len(categories):
			results[path] = (bias, len(categories))
			found = True
	
	best = None
	best_len = 0
	for k, (b, v) in results.items():
		if best:
			if b > best:
				out = k
				best_len = v
			elif b == best and v > best_len:
				out = k
				best_len = v
		else:
			best = b
			best_len = v
			out = k
		
	return out if found else None
