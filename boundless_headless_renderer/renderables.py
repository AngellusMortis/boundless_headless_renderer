def mesh_from_json(json):
    pass

class RenderableMesh():
    def __init__(self, name, arrays, material, minpos, maxpos):
        self.name = name
        self.arrays = arrays
        self.material = material
        self.minpos = minpos
        self.maxpos = maxpos

class Scene:
    def __init__(self, name, loc_base, loc_gui, loc_ammo, type_name):
        self.name = name
        self.loc_base = loc_base
        self.loc_gui = loc_gui
        self.loc_ammo = loc_ammo
        self.type_name = type_name
    
    def create_from_nodetree(self, rootname, root_data, model_json, boundlesspath):
        from pyrr import Matrix44
        transform = self.loc_gui 
        self.root = self.process_node(rootname, root_data, model_json, boundlesspath, transform)
    
    def process_node(self, name, data, model_json, boundlesspath, transform=None):
        node = Node([], name, [], transform)

        if data.get("nodes"):
            for n, v in data["nodes"].items():
                node.children.append(self.process_node(n, v, model_json, boundlesspath))
        
        if data.get("geometryinstances"):
            for n, v in data["geometryinstances"].items():
                node.meshes.append(self.process_mesh(n, v, model_json, boundlesspath))

        return node
    
    def process_mesh(self, name, data, model_json, boundlesspath):
        import numpy as np

        geometry = model_json['geometries'][data['geometry']]
        surface = geometry['surfaces'][data['surface']]
        material = model_json['materials'][data['material']]
        
        arrays = {}
        inputs = geometry['inputs']
        sources = geometry['sources']
        for semantic in inputs.keys():
            source = inputs[semantic]['source']
            offset = inputs[semantic]['offset']
            arrays[semantic] = {}
            
            arrays[semantic]["data"] = np.array(
                sources[source]['data'],
                dtype="f4"
            )

            if semantic == "POSITION":
                minpos = sources[source]["min"]
                maxpos = sources[source]["max"]
            
            index_stride = len(surface['triangles']) // (3 * surface['numPrimitives'])
            
            arrays[semantic]["indices"] = np.array(
                surface['triangles'][offset::index_stride],
                dtype="i4"
            )

        return RenderableMesh(name, arrays, material, minpos, maxpos)

    def add_ammo(self, name, data, model_json, boundlesspath):
        self.root.children.append(
            self.process_node(name, data, model_json, boundlesspath, transform=self.loc_ammo)
        )

    def calculate_bounds(self):
        import numpy as np
        from pyrr import Matrix44, Vector3
        def calculate_node_bounds(node, transform=Matrix44.identity()):
            minimum = [100, 100, 100]
            maximum = [-100, -100, -100]

            transform = transform * node.transform if isinstance(node.transform, Matrix44) else transform

            def calculate_mesh_bounds(mesh, transform):
                arrays = mesh.arrays
                pos_vertices = arrays["POSITION"]["data"]
                pos_indices = arrays["POSITION"]["indices"]
                pos_multi = np.array(np.split(pos_vertices, len(pos_vertices)//3))
                pos = pos_multi[pos_indices]
                pos = np.array([np.array([*x, 1.0]) for x in pos])
                trans_pos = np.dot(transform.T, pos.T).T
                trans_pos = np.array([x[0:3] for x in trans_pos])

                maxi = trans_pos.max(axis=0)
                mini = trans_pos.min(axis=0)

                return (
                    mini, maxi
                )

            for child in node.children:
                child_min, child_max = calculate_node_bounds(child, transform)
                minimum = np.minimum(child_min, minimum)
                maximum = np.maximum(child_max, maximum)

            for mesh in node.meshes:
                mesh_min, mesh_max = calculate_mesh_bounds(mesh, transform)
                minimum = np.minimum(mesh_min, minimum)
                maximum = np.maximum(mesh_max, maximum)
            
            return minimum, maximum

        self.bounds = calculate_node_bounds(self.root)

class Node:
    def __init__(self, children, name, meshes, transform):
        self.name = name
        self.transform = transform
        self.meshes = meshes
        self.children = children

class Renderable:
    def __init__(self, id, name, base_palette, decal_palette, type_name):
        self.id = id
        self.name = name
        self.base_palette = base_palette
        self.decal_palette = decal_palette

        self.type_name = type_name

class RenderableItem(Renderable):
    def __init__(self, id, name, base_palette, decal_palette, default_ammo, should_render_ammo, decal_texture_path, model_path, categories):
        super().__init__(id, name, base_palette, decal_palette, "ITEM")
        
        self.default_ammo = default_ammo
        self.should_render_ammo = should_render_ammo
        self.decal_texture_path = decal_texture_path
        self.model_path = model_path
        self.categories = categories

class BlockTexture:
    def __init__(self, id, special):
        self.id = id
        self.special = special

class RenderableBlock(Renderable):
    def __init__(self, id, name, base_palette, decal_palette, tex_atlas, top_tex, side_tex, top_decals, side_decals):
        super().__init__(id, name, base_palette, decal_palette, "BLOCK")

        self.tex_atlas = tex_atlas
        self.top_tex = top_tex
        self.side_tex = side_tex
        self.top_decals = top_decals
        self.side_decals = side_decals

class RenderableProp(Renderable):
    def __init__(self, id, name, base_palette, decal_palette, decal_texture_path, model_path, categories):
        super().__init__(id, name, base_palette, decal_palette, "PROP")

        self.decal_texture_path = decal_texture_path
        self.model_path = model_path
        self.categories = categories
