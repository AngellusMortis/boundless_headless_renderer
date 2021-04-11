from boundless_headless_renderer.renderer import BoundlessRenderer
import moderngl, os
import argparse

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def main():
	parser = argparse.ArgumentParser(description='Boundless Icon Renderer by @willcrutchley')
	parser.add_argument("-s", "--style", choices=["uniform", "greedy"], default="greedy", required=False)
	parser.add_argument("-r", "--resolution", required=True, help="Specify the render resolution (e.g 256 for a 256x256 image)")
	parser.add_argument("-a", "--anti-alias", action="store_true", help="Renders at a slightly higher resolution and then downscales (helps to clean up edges)")
	parser.add_argument("-o", "--overwrite", action="store_true", help="Renders all specified objects even if their image already exists in the output directory")
	parser.add_argument("--force-foliage", action="store_true", help="Forces the rendering of blocks with treeFoliage (disabled by default since the script does not correctly render these)")
	parser.add_argument("-q", "--quiet", action="store_true", help="No printing")
	parser.add_argument("-g", "--boundless-path", required=True, help="Path to MacOS Boundless install", type=dir_path)

	render_group = parser.add_argument_group(title="What to render", description="Pick a specific ID or NAME or one or more of [--items, --blocks, --props]")
	render_override = render_group.add_mutually_exclusive_group()
	render_override.add_argument("--id", help="Specify an ID to render (find it in compileditems or compiledblocks)")
	render_override.add_argument("--name", help="Specify a NAME to render (find it in compileditems or compiledblocks). May end up rendering the DUGUP version")
	render_group.add_argument("-i", "--items", action="store_true", help="Render all items")
	render_group.add_argument("-p", "--props", action="store_true", help="Render all props")
	render_group.add_argument("-b", "--blocks", action="store_true", help="Render all blocks")

	colour_group = parser.add_argument_group(title="Colouration",
		description=
		"""
		By default the script will render all colour/decal colour combinations.
		This can take a long time for blocks with 255 possible base colours and 255 possible decal colours.
		\nTherefore you can specify a specific base or decal colour to only render that one (this doesn't apply to anything using a palette with <255 options)
		"""
	)
	colour_group.add_argument("-dc", "--decal-colour", "--decal-color")
	colour_group.add_argument("-bc", "--base-colour", "--base-color")

	args = parser.parse_args()
	argvars = vars(args)
	if not (args.items or args.props or args.blocks) and not (args.id or args.name):
		parser.error('Specify at least one type of object to render (--items, --props, --blocks)')

	os.environ["BOUNDLESS_PATH"] = argvars["boundless_path"]

	ctx = moderngl.create_context(standalone=True)

	renderer = BoundlessRenderer(os.environ["BOUNDLESS_PATH"], os.path.dirname(__file__), ctx, argvars)
	scenes = {}
	if argvars["items"] or argvars["id"] or argvars["name"]:
		items = renderer.discover_items()
		item_scenes = renderer.scenes_from_renderables(items)
		scenes.update(item_scenes)
	if argvars["props"] or argvars["id"] or argvars["name"]:
		props = renderer.discover_props()
		prop_scenes = renderer.scenes_from_renderables(props)
		scenes.update(prop_scenes)
	if argvars["blocks"] or argvars["id"] or argvars["name"]:
		blocks = renderer.discover_blocks()
		block_scenes = renderer.scenes_from_renderables(blocks)
		scenes.update(block_scenes)

	for id, scene in scenes.items():
		checkpath = "out/" + scene.name
		# Naively assume that if the scene's directory exists and isn't
		# empty we can ignore it
		if os.path.exists(checkpath) and os.listdir(checkpath) and not argvars["overwrite"]:
			print("[INFO] Skipping {}".format(scene.name))
		else:
			base_palette = renderer.palette_json[scene.renderable.base_palette]
			base_ids = base_palette['colorVariations'] if base_palette['colorVariations'] else [0]
			decal_palette = renderer.palette_json[scene.renderable.decal_palette]
			decal_ids = decal_palette['colorVariations'] if decal_palette['colorVariations'] else [0]

			if scene.renderable.type_name == "PROP" and not scene.renderable.decal_texture_path: decal_ids = [0]

			base_override = decal_override = False
			if len(base_ids) == 255 and argvars["base_colour"]:
				base_ids = [argvars["base_colour"]]
				base_override = True
			if len(decal_ids) == 255 and argvars["decal_colour"]:
				decal_ids = [argvars["decal_colour"]]
				decal_override = True

			for b in range(0, len(base_ids)):
				for d in range(0, len(decal_ids)):
					renderer.render_scene(scene, b if not base_override else int(argvars["base_colour"]), d if not decal_override else int(argvars["decal_colour"]))



if __name__ == "__main__":
	main()
