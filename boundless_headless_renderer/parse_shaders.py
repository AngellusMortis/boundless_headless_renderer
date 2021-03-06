import json
import os

SHADER_FILES = [
	"voxel.tzfx.json",
	"skylight.tzfx.json",
]


class ShaderTechnique():
	def __init__(self, vertex, fragment, params, state):
		self.vertex = vertex
		self.fragment = fragment
		self.params = params
		self.state = state

def get_shaders():
	base_dir = os.path.join(os.environ["BOUNDLESS_PATH"], 'assets/shaders/macosx')
	shaders = {}
	for filename in SHADER_FILES:
		with open(os.path.join(base_dir, filename)) as shader_file:
			shader_json = json.load(shader_file)
			techniques = shader_json['techniques']
			for ident, tech in techniques.items():
				if not tech['states'].get("DepthFunc"): continue
				vertex, fragment = tech['programs']
				obj = ShaderTechnique(
					shader_json['programs'][vertex]['glsl_150'],
					"#version 140\nconst float SRGB_INVERSE_GAMMA = 2.2;\nvec4 srgb_to_rgb_approx(vec4 rgb) {\nreturn vec4(pow(rgb.xyz, vec3(SRGB_INVERSE_GAMMA)), rgb.w);\n}\n" + shader_json['programs'][fragment]['glsl_150'].replace("#version 140", "")[0:-4] + ';\nout_FragData[0]=srgb_to_rgb_approx(out_FragData[0]);\n}\n',
					tech['parameters'],
					tech['states']
				)
				shaders[ident] = obj
	return shaders
