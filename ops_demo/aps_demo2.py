"""
— Question : “What angles should I go to for WSe2 002 Bragg peak?”

Demo 2 (medium):
Step 1: call to Spec to get energy
Step 2: call materials project to get lattice constant + angles
Step 3: Run through Sector 7 calculator
Step 4: Output to user

"""

material_formula = 'WSe2'
bragg_peak = '002'


lcrc_ssh_config = {
    'hostname': 'bebop.lcrc.anl.gov',
    'port': 22,
    'username': open('./SSH_USERNAME').read().strip(),
    'pkey': './LCRC_SSH_KEY',  # ssh-keygen -t rsa
}

aps_ssh_config = {
    'hostname': '164.54.128.24',
    'port': 22,
    'username': open('./SSH_USERNAME').read().strip(),
    'pkey': './APS_SSH_KEY',  # ssh-keygen -t rsa
}

MP_API_KEY = open('./MP_API_KEY').read().strip()


# ----------------------------------------

import func

print("What angles should I go to for WSe2 002 Bragg peak?")

stdout = func.jump_ssh_run_command(
    lcrc_ssh_config,
    aps_ssh_config,
    "python -c \"import epics; print(epics.caget('26idbDCM:sm8.RBV'))\"",
)
energy = float(stdout) / 1000.0  # keV
print(f'\n# --- get energy from Epics ---\nenergy = {energy} keV')

lattice = func.mp_get_lattice(material_formula, MP_API_KEY)
print(f'\n# --- get lattice from Materials Project ---\n{lattice}')

result = func.aps7id_calculate_angle(energy, bragg_peak, lattice)
print(f'\n# --- 7id reflective calculator ---\n{result}')

print(f"\nangle = {result['angle']}")
