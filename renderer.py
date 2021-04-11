import moderngl, os, json
from wand.image import Image
from math import tan, radians
FOV = 60

import utils, renderables

import numpy as np
import parse_shaders

from pyrr import Matrix44, Vector3

class BoundlessRenderer():
	def __init__(self, boundlesspath, ctx, args):
		# Path to a boundless installation
		self.boundlesspath = boundlesspath
		self.args = args

		with open(self.boundlesspath + '/assets/archetypes/compiledcolorpalettelists.msgpack', 'rb') as palettefile:
			self.palette_json = utils.convert_msgpackfile(palettefile)

		self.lookat = Matrix44.look_at((10, 10, 0.0), (10, 10, -5), (0, 1, 0))
		self.perspective = Matrix44.perspective_projection(FOV, 1.0, 0.01, 1000.0)

		self.projection_mat = self.perspective * self.lookat
		self.model_mat = Matrix44.identity()

		# Parsed shaders
		self.shaders = parse_shaders.get_shaders()

		with open(self.boundlesspath + "/assets/archetypes/compiledspecialmaterials.msgpack", 'rb') as specialsfile:
			self.specials_json = utils.convert_msgpackfile(specialsfile)
		
		self.specials_names = list(map(lambda a: a["name"], self.specials_json))

		# A map of all the textures we use which are constant
		# Key is the uniform name, value is the texture object
		self.local_tex_path = 'assets/textures/'
		self.const_tex = {}
		# Dynamic textures
		self.dyn_tex = {}

		self.ctx = ctx
		# Necessary to stop memory leaks
		self.ctx.gc_mode = "auto"
		# Note: this is the dimensions of the image. Certain items/blocks/props won't fill
		# 	this canvas.
		self.target_size = (int(args["resolution"]), int(args["resolution"]))
		self.render_size = (
			int(self.target_size[0] // 0.9),
			int(self.target_size[1] // 0.9)
		) if args["anti_alias"] else self.target_size
		# Initialise properly later, just allocating the field
		self.prog = {}
		self.cbo = self.ctx.renderbuffer(self.render_size)
		self.dbo = self.ctx.depth_texture(self.render_size, alignment=1)
		self.fbo = self.ctx.framebuffer(color_attachments=(self.cbo), depth_attachment=self.dbo)
		self.fbo.use()

		# Initialise all of the constant textures
		self.init_constant_tex2ds()
		self.init_constant_texcubes()

		self.buffer_cache = []

		# Grab uniforms
		with open('assets/shader_dump.json') as uniformsfile:
			self.uniforms_json = json.load(uniformsfile)
		
	def lookat_closest(self, bounds, meshpos, midvec):
		if self.args["style"] == "greedy":
			mini = bounds[0] - midvec
			maxi = bounds[1] - midvec
			x_length = abs(mini[0]) + abs(maxi[0])
			y_length = abs(mini[1]) + abs(maxi[1])
			length = max(x_length, y_length)/2
			max_dist = (length / tan(radians(FOV/2))) + abs(mini[2])
			campos = Vector3([*meshpos.xy, max_dist + meshpos.z])
			self.lookat = Matrix44.look_at(campos, meshpos, (0, 1, 0))
			self.projection_mat = self.perspective * self.lookat

			projection_arr = self.projection_mat.flatten()
			projection_trans = np.array(np.hsplit(np.array(np.split(np.array(list(projection_arr[0:4]) + list(projection_arr[4:8]) + list(projection_arr[8:12]) + list(projection_arr[12:16])), 4)), 4))
			self.prog['projectionTransform'].write(projection_trans.astype('f4'))
	
	def set_prog(self, prog_name):
		technique = self.shaders[prog_name]
		self.prog = self.ctx.program(
			vertex_shader=technique.vertex,
			fragment_shader=technique.fragment,
		)

		self.ctx.depth_func = "<="

		if technique.state["BlendEnable"]:
			self.ctx.enable(self.ctx.BLEND)
			self.ctx.blend_equation = (moderngl.FUNC_ADD, moderngl.FUNC_ADD)
			self.ctx.blend_func = (
				moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA,
				moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
			)
		else:
			self.ctx.disable(self.ctx.BLEND)
		
		self.ctx.enable(self.ctx.DEPTH_TEST | self.ctx.CULL_FACE)
		
	def init_constant_tex2ds(self):
		self.const_tex2d_data = [
			# Generic
			('white.dds', "cloudShadowTexture", None),
			("white.dds", "hdrTexture", None),

			# Atlas specific
			("white.dds", "diffuseTexture", None)
		]

		for (filename, uniform_name, internal_fmt) in self.const_tex2d_data:
			self.init_tex_2d(self.local_tex_path + filename, uniform_name, internal_format=internal_fmt)
	
	def init_constant_texcubes(self):
		self.const_texcube_data = [
			# Generic
			("grey_cube", "fragment_reflection_env_map"),
		]

		for (dir, uniform_name) in self.const_texcube_data:
			self.init_tex_cube(self.local_tex_path + dir, uniform_name)
		
	def load_const_uniforms(self):
		from collections.abc import Iterable
		
		self.add_constant_tex_uniforms()
		for u_name, u_val in self.uniforms_json.items():
			val = u_val['Value']
			if self.prog.get(u_name, default=None) and val:
				if isinstance(val, Iterable):
					self.prog[u_name].write(np.array(val).astype('f4'))
				else:
					self.prog[u_name].value = val

		projection_arr = self.projection_mat.flatten()
		projection_trans = np.array(np.hsplit(np.array(np.split(np.array(list(projection_arr[0:4]) + list(projection_arr[4:8]) + list(projection_arr[8:12]) + list(projection_arr[12:16])), 4)), 4))
		self.prog['projectionTransform'].write(projection_trans.astype('f4'))
		ident_arr = Matrix44.identity().astype('f4').flatten()
		ident_43 = np.array(np.hsplit(np.array(np.split(np.array(list(ident_arr[0:3]) + list(ident_arr[4:7]) + list(ident_arr[8:11]) + list(ident_arr[12:15])), 4)), 3))
		self.prog['voxelToWorldTransform'].write(ident_43.astype('f4'))
		self.prog['worldToVoxelTransform'].write(ident_43.astype('f4'))
		self.prog['worldToViewTransform'].write(ident_43.astype('f4'))
		self.prog['viewToWorldTransform'].write(ident_43.astype('f4'))

	def load_dyn_uniforms(self):
		self.add_dyn_tex_uniforms()

		modeltoworld_arr = self.model_mat.astype('f4').flatten()
		modeltoworld_43 = np.array(np.hsplit(np.array(np.split(np.array(list(modeltoworld_arr[0:3]) + list(modeltoworld_arr[4:7]) + list(modeltoworld_arr[8:11]) + list(modeltoworld_arr[12:15])), 4)), 3))
		self.prog['modelToWorldTransform'].write(modeltoworld_43.astype('f4'))
		if self.prog.get("modelToViewTransform", default=None):
			self.prog['modelToViewTransform'].write(modeltoworld_43.astype('f4'))
			self.prog['modelToVoxelTransform'].write(modeltoworld_43.astype('f4'))

	def add_constant_tex_uniforms(self):
		for uniform_name, tex in self.const_tex.items():
			if self.prog.get(uniform_name, default=None):
				# Texture is used in the current shader
				self.prog[uniform_name].value = self.prog[uniform_name].location
				tex.use(self.prog[uniform_name].location)
	
	def add_dyn_tex_uniforms(self):
		for uniform_name, tex in self.dyn_tex.items():
			if self.prog.get(uniform_name, default=None):
				# Texture is used in the current shader
				self.prog[uniform_name].value = self.prog[uniform_name].location
				tex.use(self.prog[uniform_name].location)
	
	def init_tex_2d(self, path, uniform_name, from_boundlessdir=False, constant=True, internal_format=None):
		# Some 2D texture imports will be from the local boundless installation
		# E.g diffuse maps, specular maps etc
		if from_boundlessdir: path = "/assets/".join([self.boundlesspath, path])

		im_data = Image(filename=path)
		if "shCoef" in uniform_name:
			tex = self.ctx.texture3d((*im_data.size, 1), 4, data=bytes(im_data.export_pixels()))
		else:
			tex = self.ctx.texture(im_data.size, 4, data=bytes(im_data.export_pixels()), internal_format=internal_format)
		tex.build_mipmaps()

		if constant:
			# If we are overwriting a texture then there's an issue
			assert not self.const_tex.get(uniform_name)
			self.const_tex[uniform_name] = tex
		else:
			self.dyn_tex[uniform_name] = tex
	
	def init_block_tex_2d(self, toppath, sidepath, uniform_name, internal_format=None):
		toppath = "/assets/".join([self.boundlesspath, toppath]) if toppath else toppath
		sidepath = "/assets/".join([self.boundlesspath, sidepath]) if sidepath else sidepath

		# Unfortunately we need to use pillow for this because
		# 	wand is far too slow
		from PIL import Image as PImage
		side_im_data = PImage.open(sidepath) if sidepath else None
		top_im_data = PImage.open(toppath) if toppath else None
		cube_im_data = utils.gen_cmap(top_im_data, side_im_data)

		tex = self.ctx.texture(cube_im_data.size, 4, data=cube_im_data.tobytes(), internal_format=internal_format)
		tex.build_mipmaps()

		self.dyn_tex[uniform_name] = tex
	
	def init_tex_cube(self, dir, uniform_name, constant=True):
		"""
		The different surfaces of the cubemap
		are stored as seperate 2D images. These have names of format:
			'main surface ([direction] [axis])'
		Where direction can either be 'positive' or 'negative' and axis is 'x', 'y' or 'z'
		"""

		cubemap = b''
		# All possible cubemap surfaces, correctly indexed
		cube_files = [
			"positive x",
			"negative x",

			"positive y",
			"negative y",

			"positive y",
			"negative y",
		]

		surfacesize = None

		for surface_name in cube_files:
			surface = Image(filename=(dir + '/' + "main surface ({}).dds".format(surface_name)))
			cubemap += bytes(surface.export_pixels())
			if not surfacesize: surfacesize = surface.size
		
		tex = self.ctx.texture_cube(surfacesize, 4, data=cubemap)

		if constant:
			# If we are overwriting a texture then there's an issue
			assert not self.const_tex.get(uniform_name)
			self.const_tex[uniform_name] = tex
		else:
			self.dyn_tex[uniform_name] = tex
	
	BLOCK_MODEL_PATH = "locator_templates/models/template_block.dae"
	def scenes_from_renderables(self, r):
		from pyrr import Matrix44

		def matrix_from_loc(data):
			if not data: return None
			data = data.get("matrix")
			return Matrix44(
				[data[0:3] + [0.],
				data[3:6] + [0.],
				data[6:9] + [0.],
				data[9:12] + [1.]]
			)

		scenes = {}

		for id, renderable in r.items():
			model_path = renderable.model_path if renderable.type_name != "BLOCK" else self.BLOCK_MODEL_PATH
			with open(
				self.boundlesspath + "/assets/{}.msgpack"
				.format(model_path), 'rb'
			) as model_file:
				model_json = utils.convert_msgpackfile(model_file)

			model_json["nodes"] = {k: v for k, v in model_json["nodes"].items() if v.get('geometryinstances')}

			assert len(model_json["nodes"]) == 1
			root_node = list(model_json["nodes"].keys())[0]

			loc_base = loc_gui = loc_ammo = None

			loc_base_key = loc_gui_key = loc_ammo_key = None
			if model_json["nodes"][root_node].get("nodes"):
				for key in list(model_json["nodes"][root_node]["nodes"].keys()):
					if "loc_gui" in key: loc_gui_key = key
					if "loc_ammo" in key: loc_ammo_key = key
					if "loc_base" in key: loc_base_key = key
				loc_base = matrix_from_loc(
					model_json["nodes"][root_node]["nodes"].get(loc_base_key)
				)
				loc_gui = matrix_from_loc(
					model_json["nodes"][root_node]["nodes"].get(loc_gui_key)
				)
				loc_ammo = matrix_from_loc(
					model_json["nodes"][root_node]["nodes"].get(loc_ammo_key)
				)
			
			locator_path = utils.get_locator(renderable.categories, self.boundlesspath) if renderable.type_name != "BLOCK" else model_path
			if not locator_path and not self.args["quiet"]: print("[WARN] Missing locator template for {}".format(renderable.name))
			else:
				with open(
					self.boundlesspath + "/assets/{}.msgpack"
					.format(locator_path), 'rb'
				) as loc_file:
					loc_json = utils.convert_msgpackfile(loc_file)
			
				assert len(loc_json["nodes"]) == 1
				loc_root_node = list(loc_json["nodes"].keys())[0]
				if loc_json["nodes"][loc_root_node].get("nodes"):
					for key in list(loc_json["nodes"][loc_root_node]["nodes"].keys()):
						if "loc_gui" in key: loc_gui_key = key
						if "loc_ammo" in key: loc_ammo_key = key
						if "loc_base" in key: loc_base_key = key
					if not isinstance(loc_base, Matrix44):
						loc_base = matrix_from_loc(
							loc_json["nodes"][loc_root_node]["nodes"].get(loc_base_key)
						)
					if not isinstance(loc_gui, Matrix44):
						loc_gui = matrix_from_loc(
							loc_json["nodes"][loc_root_node]["nodes"].get(loc_gui_key)
						)
					if not isinstance(loc_ammo, Matrix44):
						loc_ammo = matrix_from_loc(
							loc_json["nodes"][loc_root_node]["nodes"].get(loc_ammo_key)
						)

			if not isinstance(loc_gui, Matrix44): loc_gui = Matrix44.identity()
			assert(isinstance(loc_gui, Matrix44))

			scene = renderables.Scene(renderable.name, loc_base, loc_gui, loc_ammo, renderable.type_name)
			scene.create_from_nodetree(root_node, model_json["nodes"][root_node], model_json, self.boundlesspath)
			scene.renderable = renderable

			if renderable.type_name == "ITEM" and renderable.should_render_ammo and renderable.default_ammo: # Only items can have default_ammo
				ammo_path = renderable.default_ammo
				with open(
					self.boundlesspath + "/assets/{}.msgpack"
					.format(ammo_path), 'rb'
				) as ammo_file:
					ammo_json = utils.convert_msgpackfile(ammo_file)

				assert(len(ammo_json["nodes"].keys()) == 1)
				ammo_root = list(ammo_json["nodes"].keys())[0]
				assert(ammo_json["nodes"][ammo_root].get("geometryinstances"))
				
				scene.add_ammo(ammo_root, ammo_json["nodes"][ammo_root], ammo_json, self.boundlesspath)
		
			scene.calculate_bounds()
			scenes[id] = scene

		return scenes

	def render_scene(self, scene, basepaletteindex, decalpaletteindex):
		if scene.type_name == "BLOCK":
			self.deferred_cutout = []
			self.deferred_decal = []
			self.render_scene_blocks(scene, basepaletteindex, decalpaletteindex)
		else:
			self.deferred_alpha = []
			self.render_scene_itemsprops(scene, basepaletteindex, decalpaletteindex)

	def render_scene_blocks(self, scene, basepaletteindex, decalpaletteindex):
		if not self.args["quiet"]: print("[INFO] " + scene.name + " (b: {}, d: {})".format(basepaletteindex, decalpaletteindex))
		self.ctx.clear(0.0, 0.0, 0.0, 0.0, 1.0)

		assert(len(scene.root.meshes) == 1)
		blockmesh = scene.root.meshes[0]
		blockmesh_name = scene.root.meshes[0].name
		arrays = blockmesh.arrays
		pos_vertices = arrays["POSITION"]["data"]
		pos_indices = arrays["POSITION"]["indices"]
		uv_vertices = arrays["TEXCOORD0"]["data"]
		uv_indices = arrays["TEXCOORD0"]["indices"]
		norm_vertices = arrays["NORMAL"]["data"]
		norm_indices = arrays["NORMAL"]["indices"]
		binorm_vertices = arrays["BINORMAL"]["data"]
		binorm_indices = arrays["BINORMAL"]["indices"]
		tang_vertices = arrays["TANGENT"]["data"]
		tang_indices = arrays["TANGENT"]["indices"]
		
		pos_multi = np.array(np.split(pos_vertices, len(pos_vertices)//3))
		pos_monolith = pos_multi[pos_indices].flatten()
		blk_pos_buf = self.ctx.buffer(pos_monolith.astype('f4').tobytes())
		
		uv_multi = np.array(np.split(uv_vertices, len(uv_vertices)//2))
		uv_monolith = uv_multi[uv_indices].flatten()
		blk_uv_buf = self.ctx.buffer(uv_monolith.astype('f4').tobytes())
		
		norm_multi = np.array(np.split(norm_vertices, len(norm_vertices)//3))
		norm_monolith = norm_multi[norm_indices].flatten()
		blk_norm_buf = self.ctx.buffer(norm_monolith.astype('f4').tobytes())
		
		binorm_multi = np.array(np.split(binorm_vertices, len(binorm_vertices)//3))
		binorm_monolith = binorm_multi[binorm_indices].flatten()
		blk_binorm_buf = self.ctx.buffer(binorm_monolith.astype('f4').tobytes())
		
		tang_multi = np.array(np.split(tang_vertices, len(tang_vertices)//3))
		tang_monolith = tang_multi[tang_indices].flatten()
		blk_tang_buf = self.ctx.buffer(tang_monolith.astype('f4').tobytes())

		def load_block_material(renderer, renderable, node):
			expected = {
				"diffuse": None,
				"gradient_mask": None,
				"normal": None,
				"sm_em_mt": None,
			}

			file_prefixes = {
				"diffuse": "cm",
				"gradient_mask": "gm_mm",
				"normal": "nm",
				"sm_em_mt": "sm_em_mt"
			}

			atlases = {
				"blocks": "blockatlas",
				"specialMaterialBlocks": "specialmaterialsatlas",
				"alphaBlocks": "alphablockatlas"
			}

			atlaspath = atlases[renderable.tex_atlas]
			if renderable.tex_atlas == "specialMaterialBlocks" or renderable.top_tex.special:
				renderable.top_tex.id = self.specials_json[renderable.top_tex.id]["textures"][0]
				atlaspath = "specialmaterialsatlas"
			if renderable.tex_atlas == "specialMaterialBlocks" or renderable.side_tex.special:
				renderable.side_tex.id = self.specials_json[renderable.side_tex.id]["textures"][0]
				atlaspath = "specialmaterialsatlas"

			for uniform_name, fmt in expected.items():
				file_prefix = file_prefixes[uniform_name]
				top_path = "textures/{}/{}_{}.dds".format(atlaspath, file_prefix, renderable.top_tex.id)
				side_path = "textures/{}/{}_{}.dds".format(atlaspath, file_prefix, renderable.side_tex.id)
				self.init_block_tex_2d(top_path, side_path, uniform_name, internal_format=fmt)
			self.init_block_tex_2d(None, None, "decal_gradient_mask")
	
			assert(len(renderable.top_decals) == len(renderable.side_decals))
			for top_decal, side_decal in zip(renderable.top_decals, renderable.side_decals):
				atlaspath = atlases[renderable.tex_atlas]
				if top_decal.special:
					top_decal.id = self.specials_json[top_decal.id]["textures"][0]
					atlaspath = "specialmaterialsatlas"
				if not top_decal.special: atlaspath = "alphablockatlas"
				if side_decal.special:
					side_decal.id = self.specials_json[side_decal.id]["textures"][0]
					atlaspath = "specialmaterialsatlas"
				if not side_decal.special: atlaspath = "alphablockatlas"
				
				from copy import deepcopy
				decal_node = renderables.Node([], node.name + "_decal", [deepcopy(node.meshes[0])], node.transform)
				decal_cutout_node = renderables.Node([], node.name + "_decalcut", [deepcopy(node.meshes[0])], node.transform)
				decal_node.meshes[0].material = {
					"parameters": {
					},
					"effect": "blinn_alpha"
				}
				decal_cutout_node.meshes[0].material = {
					"parameters": {
					},
					"effect": "blinn_alphaCutout"
				}
				for uniform_name, _ in expected.items():
					file_prefix = file_prefixes[uniform_name]
					top_path = "textures/{}/{}_{}.dds".format(atlaspath, file_prefix, top_decal.id)
					side_path = "textures/{}/{}_{}.dds".format(atlaspath, file_prefix, side_decal.id)
					decal_node.meshes[0].material["parameters"][uniform_name] =  {
						"top": top_path,
						"side": side_path
					}
					decal_cutout_node.meshes[0].material["parameters"][uniform_name] =  {
						"top": top_path,
						"side": side_path
					}
				self.deferred_decal.append(decal_node)
				self.deferred_decal.append(decal_cutout_node)
		
		def load_decal_material(renderer, material):
			expected = {
				"diffuse": None,
				"gradient_mask": None,
				"normal": None,
				"sm_em_mt": None,
			}

			for uniform_name, fmt in expected.items():
				top_path = material["parameters"][uniform_name]["top"]
				side_path = material["parameters"][uniform_name]["side"]
				self.init_block_tex_2d(top_path, side_path, uniform_name, internal_format=fmt)
			self.init_block_tex_2d(None, None, "decal_gradient_mask")

		# This is setup to support meshes other than a block mesh.
		# 	I naively thought I'd be able to support foliage this way, which isn't the case.
		def render_node(node, transform=Matrix44.identity(), decalpass=False, cutout=False):
			transform = transform * node.transform if isinstance(node.transform, Matrix44) else transform
			for child in node.children:
				render_node(child, transform)
			for mesh in node.meshes:
				prog_name = mesh.material["effect"]

				if scene.renderable.tex_atlas == "alphaBlocks" and not decalpass and not cutout:
					prog_name = "blinn_alpha"
					from copy import deepcopy
					cutout_node = renderables.Node([], node.name + "_cut", [deepcopy(node.meshes[0])], node.transform)
					self.deferred_cutout.append(cutout_node)
				elif scene.renderable.tex_atlas == "alphaBlocks" and cutout and not decalpass:
					prog_name = "blinn_alphaCutout"

				self.set_prog(prog_name)
				self.load_const_uniforms()

				if mesh.name == blockmesh_name and not decalpass:
					load_block_material(self, scene.renderable, node)
				elif mesh.name == blockmesh_name:
					load_decal_material(self, mesh.material)

				midvec = (scene.bounds[0] + scene.bounds[1]) / 2
				meshpos = Vector3([10.0, 10.0, -5.0])
				self.lookat_closest(scene.bounds, meshpos, midvec)

				self.model_mat = Matrix44.from_translation(meshpos) * Matrix44.from_translation(-midvec) * transform
				self.load_dyn_uniforms()

				if not decalpass:
					blinn_gradient = [0, 0, 0]
					palette_id = scene.renderable.base_palette
					palette = self.palette_json[palette_id]
					if palette['colorVariations']:
						gradient = palette['colorVariations'][basepaletteindex][0]
						blinn_gradient[0] = utils.correctDecimal(gradient[0])
						blinn_gradient[1] = utils.correctDecimal(gradient[1])
						blinn_gradient[2] = utils.correctDecimal(gradient[2])
					if self.prog.get("blinn_gradient", default=None):
						self.prog['blinn_gradient'].write(np.array(blinn_gradient).astype('i4'))
				if decalpass:
					blinnd_gradient = [0, 0, 0]
					palette_id = scene.renderable.decal_palette
					palette = self.palette_json[palette_id]
					if palette['colorVariations']:
						gradient = palette['colorVariations'][decalpaletteindex][0]
						blinnd_gradient[0] = utils.correctDecimal(gradient[0])
						blinnd_gradient[1] = utils.correctDecimal(gradient[1])
						blinnd_gradient[2] = utils.correctDecimal(gradient[2])
					if self.prog.get("blinn_gradient", default=None):
						self.prog['blinn_gradient'].write(np.array(blinnd_gradient).astype('i4'))

				for name, bufs in self.buffer_cache:
					if name == mesh.name:
						vao = self.ctx.simple_vertex_array(self.prog, bufs["POSITION"], "ATTR0")
						vao.bind(0, 'f', bufs["POSITION"], '3f4')
						vao.bind(1, 'f', bufs["NORMAL"], '3f4')
						vao.bind(2, 'f', bufs["TANGENT"], '3f4')
						vao.bind(3, 'f', bufs["BINORMAL"], '3f4')
						vao.bind(4, 'f', bufs["TEXCOORD0"], '2f4')
						break
				if not mesh.name == blockmesh_name:
					arrays = mesh.arrays
					pos_vertices = arrays["POSITION"]["data"]
					pos_indices = arrays["POSITION"]["indices"]
					uv_vertices = arrays["TEXCOORD0"]["data"]
					uv_indices = arrays["TEXCOORD0"]["indices"]
					norm_vertices = arrays["NORMAL"]["data"]
					norm_indices = arrays["NORMAL"]["indices"]
					binorm_vertices = arrays["BINORMAL"]["data"]
					binorm_indices = arrays["BINORMAL"]["indices"]
					tang_vertices = arrays["TANGENT"]["data"]
					tang_indices = arrays["TANGENT"]["indices"]
					
					pos_multi = np.array(np.split(pos_vertices, len(pos_vertices)//3))
					pos_monolith = pos_multi[pos_indices].flatten()
					pos_buf = self.ctx.buffer(pos_monolith.astype('f4').tobytes())
					
					uv_multi = np.array(np.split(uv_vertices, len(uv_vertices)//2))
					uv_monolith = uv_multi[uv_indices].flatten()
					uv_buf = self.ctx.buffer(uv_monolith.astype('f4').tobytes())
					
					norm_multi = np.array(np.split(norm_vertices, len(norm_vertices)//3))
					norm_monolith = norm_multi[norm_indices].flatten()
					norm_buf = self.ctx.buffer(norm_monolith.astype('f4').tobytes())
					
					binorm_multi = np.array(np.split(binorm_vertices, len(binorm_vertices)//3))
					binorm_monolith = binorm_multi[binorm_indices].flatten()
					binorm_buf = self.ctx.buffer(binorm_monolith.astype('f4').tobytes())
					
					tang_multi = np.array(np.split(tang_vertices, len(tang_vertices)//3))
					tang_monolith = tang_multi[tang_indices].flatten()
					tang_buf = self.ctx.buffer(tang_monolith.astype('f4').tobytes())

					vao = self.ctx.simple_vertex_array(self.prog, pos_buf, "ATTR0")
					vao.bind(0, 'f', pos_buf, '3f4')
					vao.bind(1, 'f', norm_buf, '3f4')
					vao.bind(2, 'f', tang_buf, '3f4')
					vao.bind(3, 'f', binorm_buf, '3f4')
					vao.bind(4, 'f', uv_buf, '2f4')
				else:
					vao = self.ctx.simple_vertex_array(self.prog, blk_pos_buf, "ATTR0")
					vao.bind(0, 'f', blk_pos_buf, '3f4')
					vao.bind(1, 'f', blk_norm_buf, '3f4')
					vao.bind(2, 'f', blk_tang_buf, '3f4')
					vao.bind(3, 'f', blk_binorm_buf, '3f4')
					vao.bind(4, 'f', blk_uv_buf, '2f4')
				
				vao.render(mode=moderngl.TRIANGLES)

				if not cutout and not decalpass:
					for n in self.deferred_cutout:
						render_node(n, cutout=True)
					self.deferred_cutout = []

				if not decalpass and not cutout:
					for n in self.deferred_decal:
						render_node(n, decalpass=True)
					self.deferred_decal = []
		
		render_node(scene.root)

		out = Image(width=self.render_size[0], height=self.render_size[1])
		out.import_pixels(data=self.fbo.read(components=4, attachment=0), channel_map="RGBA")
		out.flip()
		out.resize(width=self.target_size[0], height=self.target_size[1])
		dirpath = "out/" + scene.name
		os.makedirs(dirpath, exist_ok=True)
		out.save(filename=(dirpath + "/{}_{}".format(basepaletteindex, decalpaletteindex) + ".png"))

	def render_scene_itemsprops(self, scene, basepaletteindex, decalpaletteindex):
		if not self.args["quiet"]: print("[INFO] " + scene.name + " (b: {}, d: {})".format(basepaletteindex, decalpaletteindex))
		self.ctx.clear(0.0, 0.0, 0.0, 0.0, 1.0)

		def load_material(renderer, material_params, decal_texture_path):
			self.dyn_tex = {}

			if decal_texture_path and not material_params.get("decal_gradient_mask"):
				material_params["decal_gradient_mask"] = decal_texture_path

			expected = {
				"diffuse": None,
				"gradient_mask": None,
				"normal": None,
				"sm_em_mt": None,
				"decal_gradient_mask": None
			}

			for uniform_name, tex_path in material_params.items():
				if uniform_name == "specular_emissive" or uniform_name == "specular_emissive_metal":
					uniform_name = "sm_em_mt"
				renderer.init_tex_2d(tex_path, uniform_name, from_boundlessdir=True, constant=False, internal_format=expected[uniform_name])

			for name in expected.keys():
				if name not in self.dyn_tex.keys():
					from PIL import Image as PImage
					self.dyn_tex[name] = self.ctx.texture((1, 1), 4, data=PImage.new('RGBA', (1, 1), color=(0, 0, 0, 0)).tobytes())

		def handle_special_material(renderer, material, node):
			from copy import deepcopy
			assert(material.get('meta'))
			assert(material['meta'].get('special_material'))
			index = self.specials_names.index(material['meta']['special_material'])
			alpha = material['meta'].get("alpha")

			texture_id = self.specials_json[index]['textures'][0]
			texture_dir = "textures/specialmaterialsatlas/"
			node.meshes[0].material = {
				"parameters": {
					"diffuse": texture_dir + "cm_{}.dds".format(texture_id),
					"sm_em_mt": texture_dir + "sm_em_mt_{}.dds".format(texture_id),
					"normal": texture_dir + "nm_{}.dds".format(texture_id),
					"gradient_mask": texture_dir + "gm_mm_{}.dds".format(texture_id),
				},
				"effect": material['effect']
			}
			if not alpha:
				render_node(node, deferred=True)
			else: self.deferred_alpha.append(node)

			if material['meta'].get('decal_special_material'):
				decal_index = self.specials_names.index(material['meta']['decal_special_material'])
				decal_texture_id = texture_id = self.specials_json[decal_index]['textures'][0]
				decal_node = renderables.Node([], node.name + "_specialdecal", [deepcopy(node.meshes[0])], node.transform)
				decal_node.meshes[0].material = {
					"parameters": {
						"diffuse": texture_dir + "cm_{}.dds".format(decal_texture_id),
						"sm_em_mt": texture_dir + "sm_em_mt_{}.dds".format(decal_texture_id),
						"normal": texture_dir + "nm_{}.dds".format(decal_texture_id),
						"gradient_mask": texture_dir + "gm_mm_{}.dds".format(decal_texture_id),
					},
					"effect": "blinn_alpha"
				}
				decal_node.meshes[0].name += "_specialdecal"
				self.deferred_alpha.append(decal_node)

		def render_node(node, transform=Matrix44.identity(), deferred=False):
			transform = transform * node.transform if isinstance(node.transform, Matrix44) else transform
			for child in node.children:
				render_node(child, transform)
			for mesh in node.meshes:
				prog_name = mesh.material["effect"]
				self.set_prog(prog_name)
				self.load_const_uniforms()

				# If it's a material with alpha render it last
				from copy import deepcopy
				if (mesh.material.get('meta') and mesh.material["meta"].get("alpha")) and not deferred and not (mesh.material.get('meta') and mesh.material['meta'].get('special_material')):
					newnode = deepcopy(node)
					newnode.transform = transform
					newnode.children = []
					newnode.meshes = [deepcopy(mesh)]
					self.deferred_alpha.append(newnode)
					continue

				if mesh.material.get('meta') and mesh.material["meta"].get("special_material"):
					newnode = deepcopy(node)
					newnode.children = []
					newnode.transform = transform
					newnode.meshes = [deepcopy(mesh)]
					# will result in deferring the current node
					handle_special_material(self, deepcopy(mesh.material), newnode)
					continue
				else:
					load_material(self, mesh.material["parameters"], scene.renderable.decal_texture_path)

				midvec = (scene.bounds[0] + scene.bounds[1]) / 2
				meshpos = Vector3([10.0, 10.0, -5.0])

				self.lookat_closest(scene.bounds, meshpos, midvec)
				self.model_mat = Matrix44.from_translation(meshpos) * Matrix44.from_translation(-midvec) * transform
				self.load_dyn_uniforms()
				
				# The only thing which needs this is the atlas
				try:
					self.prog["modelViewOffset"].value = tuple(meshpos)
				except: pass

				blinn_gradient = [0, 0, 0]
				palette_id = scene.renderable.base_palette
				palette = self.palette_json[palette_id]
				if palette['colorVariations']:
					gradient = palette['colorVariations'][basepaletteindex][0]
					blinn_gradient[0] = utils.correctDecimal(gradient[0])
					blinn_gradient[1] = utils.correctDecimal(gradient[1])
					blinn_gradient[2] = utils.correctDecimal(gradient[2])
				if self.prog.get("blinn_gradient", default=None):
					self.prog['blinn_gradient'].write(np.array(blinn_gradient).astype('i4'))

				blinnd_gradient = [0, 0, 0]
				palette_id = scene.renderable.decal_palette
				palette = self.palette_json[palette_id]
				if palette['colorVariations']:
					gradient = palette['colorVariations'][decalpaletteindex][0]
					blinnd_gradient[0] = utils.correctDecimal(gradient[0])
					blinnd_gradient[1] = utils.correctDecimal(gradient[1])
					blinnd_gradient[2] = utils.correctDecimal(gradient[2])
				if self.prog.get("blinn_decalGradient", default=None):
					self.prog['blinn_decalGradient'].write(np.array(blinnd_gradient).astype('i4'))

				cached = False
				for name, bufs in self.buffer_cache:
					if name == mesh.name:
						vao = self.ctx.simple_vertex_array(self.prog, bufs["POSITION"], "ATTR0")
						vao.bind(0, 'f', bufs["POSITION"], '3f4')
						vao.bind(1, 'f', bufs["NORMAL"], '3f4')
						vao.bind(2, 'f', bufs["TANGENT"], '3f4')
						vao.bind(3, 'f', bufs["BINORMAL"], '3f4')
						vao.bind(4, 'f', bufs["TEXCOORD0"], '2f4')
						cached = True
						break
				if not cached:
					arrays = mesh.arrays
					pos_vertices = arrays["POSITION"]["data"]
					pos_indices = arrays["POSITION"]["indices"]
					uv_vertices = arrays["TEXCOORD0"]["data"]
					uv_indices = arrays["TEXCOORD0"]["indices"]
					norm_vertices = arrays["NORMAL"]["data"]
					norm_indices = arrays["NORMAL"]["indices"]
					binorm_vertices = arrays["BINORMAL"]["data"]
					binorm_indices = arrays["BINORMAL"]["indices"]
					tang_vertices = arrays["TANGENT"]["data"]
					tang_indices = arrays["TANGENT"]["indices"]
					
					pos_multi = np.array(np.split(pos_vertices, len(pos_vertices)//3))
					pos_monolith = pos_multi[pos_indices].flatten()
					pos_buf = self.ctx.buffer(pos_monolith.astype('f4').tobytes())
					
					uv_multi = np.array(np.split(uv_vertices, len(uv_vertices)//2))
					uv_monolith = uv_multi[uv_indices].flatten()
					uv_buf = self.ctx.buffer(uv_monolith.astype('f4').tobytes())
					
					norm_multi = np.array(np.split(norm_vertices, len(norm_vertices)//3))
					norm_monolith = norm_multi[norm_indices].flatten()
					norm_buf = self.ctx.buffer(norm_monolith.astype('f4').tobytes())
					
					binorm_multi = np.array(np.split(binorm_vertices, len(binorm_vertices)//3))
					binorm_monolith = binorm_multi[binorm_indices].flatten()
					binorm_buf = self.ctx.buffer(binorm_monolith.astype('f4').tobytes())
					
					tang_multi = np.array(np.split(tang_vertices, len(tang_vertices)//3))
					tang_monolith = tang_multi[tang_indices].flatten()
					tang_buf = self.ctx.buffer(tang_monolith.astype('f4').tobytes())

					vao = self.ctx.simple_vertex_array(self.prog, pos_buf, "ATTR0")
					vao.bind(0, 'f', pos_buf, '3f4')
					vao.bind(1, 'f', norm_buf, '3f4')
					vao.bind(2, 'f', tang_buf, '3f4')
					vao.bind(3, 'f', binorm_buf, '3f4')
					vao.bind(4, 'f', uv_buf, '2f4')

					self.buffer_cache.append((
						mesh.name,
						{
							"POSITION": pos_buf,
							"TEXCOORD0": uv_buf,
							"NORMAL": norm_buf,
							"BINORMAL": binorm_buf,
							"TANGENT": tang_buf
						}
					))
					if len(self.buffer_cache) > 6: self.buffer_cache.pop(0)

				vao.render(mode=moderngl.TRIANGLES)
			
			if len(self.deferred_alpha) > 1 and not self.args["quiet"]: print("[WARN] Deferred {} alpha nodes".format(len(self.deferred_alpha)))
			if not deferred:
				for node in self.deferred_alpha:
					render_node(node, deferred=True)
				self.deferred_alpha = []

		render_node(scene.root)

		out = Image(width=self.render_size[0], height=self.render_size[1])
		out.import_pixels(data=self.fbo.read(components=4, attachment=0), channel_map="RGBA")
		out.flip()
		out.resize(width=self.target_size[0], height=self.target_size[1])
		dirpath = "out/" + scene.name
		os.makedirs(dirpath, exist_ok=True)
		out.save(filename=(dirpath + "/{}_{}".format(basepaletteindex, decalpaletteindex) + ".png"))
	
	def discover_items(self):
		with open(self.boundlesspath + "/assets/archetypes/compileditems.msgpack", 'rb') as itemsfile:
			items_json = utils.convert_msgpackfile(itemsfile)

		discovered = {}
		
		for id, data in items_json.items():
			if data:
				id_check = (self.args["id"] and data["id"] == self.args["id"]) or not self.args["id"]
				name_check = (self.args["name"] and data["name"] == self.args["name"]) or not self.args["name"]
			if data["lodPaths"] and data["canDevGive"] and id_check and name_check: # Disregard everything we don't need
				if data["defaultAmmoType"]:
					default_ammo = items_json[str(data['defaultAmmoType'])]["lodPaths"][0]["paths"][0]
				else:
					default_ammo = None
				should_render_ammo = not data['renderOnlyAmmoInHand']
				base_palette = data['basePaletteId']
				decal_palette = data['decalPaletteId']
				decal_texture_path = data['decalTexture']
				name = data['name']
				# Only ever one model in data['lodPaths']
				# lodPaths.paths is a list of LODded models, we want LOD_0
				model_path = data['lodPaths'][0]['paths'][0]
				categories = data['categories']

				item = renderables.RenderableItem(id, name, base_palette, decal_palette, default_ammo, should_render_ammo, decal_texture_path, model_path, categories)
				if ((self.args["id"] and id == self.args["id"]) or not self.args["id"]) or ((self.args["name"] and data["name"] == self.args["name"]) or not self.args["name"]): discovered[id] = item

		return discovered
	
	def discover_props(self):
		with open(self.boundlesspath + "/assets/archetypes/compiledblocks.msgpack", 'rb') as blocksfile:
			blocks_json = utils.convert_msgpackfile(blocksfile)

		discovered = {}
		
		for data in blocks_json["BlockTypesData"]:
			if data:
				id_check = (self.args["id"] and data["id"] == self.args["id"]) or not self.args["id"]
				name_check = (self.args["name"] and data["name"] == self.args["name"]) or not self.args["name"]
			if data and data["mesh"] and data['canGive'] and id_check and name_check: # Disregard everything we don't need
				base_palette = data['basePaletteId']
				decal_palette = data['decalPaletteId']
				decal_texture_path = data["decalTextures"][0] if data["decalTextures"] else None
				name = data['name']
				model_path = data['mesh']['meshes'][0]['lodPaths'][0]['paths'][0]
				categories = data['categories']

				prop = renderables.RenderableProp(data["id"], name, base_palette, decal_palette, decal_texture_path, model_path, categories)
				if ((self.args["id"] and id == self.args["id"]) or not self.args["id"]) or ((self.args["name"] and data["name"] == self.args["name"]) or not self.args["name"]): discovered[data["id"]] = prop

		return discovered
	
	def discover_blocks(self):
		with open(self.boundlesspath + "/assets/archetypes/compiledblocks.msgpack", 'rb') as blocksfile:
			blocks_json = utils.convert_msgpackfile(blocksfile)
		
		discovered = {}
		dugups = []

		def get_block(data):
			base_palette = data['basePaletteId']
			decal_palette = data['decalPaletteId']
			name = data['name']
			tex_atlas = data["textureAtlas"]
			top_tex = renderables.BlockTexture(
				data["baseTexturing"]["faceTextureIndices"][0][0],
				data["baseTexturing"]["specialMaterial"]
			)
			side_tex = renderables.BlockTexture(
				data["baseTexturing"]["faceTextureIndices"][2][0],
				data["baseTexturing"]["specialMaterial"]
			)
			top_decals = []
			side_decals = []

			for decal_data in data["decalTexturing"]:
				if decal_data["faceTextureIndices"][0]:
					top_decals.append(renderables.BlockTexture(
						decal_data["faceTextureIndices"][0][0],
						decal_data["specialMaterial"]
					))
				if decal_data["faceTextureIndices"][2]:
					side_decals.append(renderables.BlockTexture(
						decal_data["faceTextureIndices"][2][0],
						decal_data["specialMaterial"]
					))
			
			block = renderables.RenderableBlock(data["id"], name, base_palette, decal_palette, tex_atlas, top_tex, side_tex, top_decals, side_decals)
			return block

		for data in blocks_json["BlockTypesData"]:
			if data:
				id_check = (self.args["id"] and data["id"] == self.args["id"]) or not self.args["id"]
				name_check = (self.args["name"] and data["name"] == self.args["name"]) or not self.args["name"]
			if data and data["canGive"] and (data["id"] == data["analyticsType"]) and data["id"] and not data["mesh"] and ((data["treeFoliage"] and self.args["force_foliage"]) or not data["treeFoliage"]) and id_check and name_check:
				block = get_block(data)
				discovered[data["id"]] = block
			elif data and data["dugup"] and (data["id"] == data["analyticsType"]) and id_check and name_check:
				dugups.append(data["dugup"])
		
		for data in blocks_json["BlockTypesData"]:
			if data and data["id"] in dugups and data["canGive"] and ((data["treeFoliage"] and self.args["force_foliage"]) or not data["treeFoliage"]):
				block = get_block(data)
				discovered[data["id"]] = block

		return discovered
