api ='7K906i48Jnr52lOD3uEE3MpUNfAnmP1N'
# from pymatgen.ext.matproj import MPRester

# def get_lattice(material_id, api_key):
#     with MPRester(api_key) as m:
#         data = m.get_structure_by_material_id(material_id)
#         lattice = data.lattice
#         return lattice.lengths, lattice.angles

# # Example usage:
# lengths, angles = get_lattice("mp-149", "your_correct_api_key")
# print("Lattice lengths:", lengths)
# print("Lattice angles:", angles)




from mp_api.matproj import MPRester

mpr = MPRester(api)
mpr.materials.available_fields
