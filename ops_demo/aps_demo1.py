"""
— Question : “What angles should I go to for WSe2 002 Bragg peak?”

Demo 1 (easy):
Assume energy is 10 KeV
Step 1: call materials project to get lattice constant + angles
Step 2: Run through Sector 7 calculator
Step 3: Output to user

"""

material_formula = 'WSe2'
bragg_peak = '002'
energy = 10  # keV

MP_API_KEY = open('./MP_API_KEY').read().strip()


# ----------------------------------------

import func

print("What angles should I go to for WSe2 002 Bragg peak?")

print(f'\n# --- assumption ---\nenergy = {energy} keV')

lattice = func.mp_get_lattice(material_formula, MP_API_KEY)
print(f'\n# --- get lattice from Materials Project ---\n{lattice}')

result = func.aps7id_calculate_angle(energy, bragg_peak, lattice)
print(f'\n# --- 7id reflective calculator ---\n{result}')

print(f"\nangle = {result['angle']}")
