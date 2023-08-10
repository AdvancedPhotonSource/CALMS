"""
— Question : “What angles should I go to for WSe2 002 Bragg peak?”

Demo 1 (easy):
Assume energy is 10 KeV
Step 1: call materials project to get lattice constant + angles
Step 2: Run through Sector 7 calculator
Step 3: Output to user

"""

import io

import requests
from bs4 import BeautifulSoup


def mp_get_lattice(material_formula: str, apikey: str) -> dict:
    """get lattice parameters from Materials Project"""
    url = "https://api.materialsproject.org/materials/summary/"
    kwargs = {
        'headers': {"X-API-KEY": apikey},
        'params': {
            # see https://api.materialsproject.org/docs#/Materials%20Summary
            'formula': material_formula,
            'deprecated': False,
            '_fields': ','.join(
                [
                    'material_id',
                    'formula_pretty',
                    'energy_above_hull',
                    'is_stable',
                    'theoretical',
                    'structure',
                    'symmetry',
                ]
            ),
        },
    }

    response = requests.get(url, **kwargs)
    results = response.json()['data']

    energy_sorted = sorted(
        [
            (
                # sorted by energy_above_hull
                mat['energy_above_hull'],
                # prefer stable and experimentally observed structures
                int(not mat['is_stable']) + int(mat['theoretical']),
                # original index in results
                ix,
            )
            for ix, mat in enumerate(results)
        ]
    )

    selected = results[energy_sorted[0][2]]

    symmetry = selected['symmetry']
    lattice = selected['structure']['lattice']
    info = {
        'id': selected['material_id'],
        'formula': selected['formula_pretty'],
        'symmetry': symmetry['symbol'],
        'crystal': symmetry['crystal_system'],
        'a': lattice['a'],
        'b': lattice['b'],
        'c': lattice['c'],
        'alpha': lattice['alpha'],
        'beta': lattice['beta'],
        'gamma': lattice['gamma'],
    }
    return info


def calculate_angle(energy: float, bragg_peak: str, lattice: dict) -> float:
    """run 7id reflective calculator"""
    url = 'https://7id.xray.aps.anl.gov/cgi-bin/elastic.cgi'
    data = {
        'mode': 'fe',
        'fixed': energy,
        'crystal': 'TRI',
        'lpa': lattice['a'],
        'lpalpha': lattice['alpha'],
        'lpb': lattice['b'],
        'lpbeta': lattice['beta'],
        'lpc': lattice['c'],
        'lpgamma': lattice['gamma'],
    }

    response = requests.post(url, data=data)
    page = BeautifulSoup(response.content, 'lxml')

    # extract angle
    angle = None
    for line in io.StringIO(page.text):
        sp = line.split()
        if len(sp) >= 5 and f"{''.join(sp[:3])}" == bragg_peak:
            angle = float(sp[4])
            break
    return {'enegy': energy, 'bragg_peak': bragg_peak, 'angle': angle}


if __name__ == '__main__':
    print("What angles should I go to for WSe2 002 Bragg peak?")

    material_formula = 'WSe2'
    bragg_peak = '002'
    energy = 10  # keV

    MP_API_KEY = open('./MP_API_KEY').read().strip()

    lattice = mp_get_lattice(material_formula, MP_API_KEY)
    print('\n# --- get lattice from Materials Project ---\n', lattice)

    result = calculate_angle(energy, bragg_peak, lattice)
    print('\n# --- 7id reflective calculator ---\n', result)
