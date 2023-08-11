import io

import paramiko
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


def aps7id_calculate_angle(
    energy: float, bragg_peak: str, lattice: dict
) -> float:
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


def jump_ssh_run_command(jump_config: dict, target_config: dict, command: str):
    with paramiko.SSHClient() as jump, paramiko.SSHClient() as target:
        for ssh in (jump, target):
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        for config in (jump_config, target_config):
            if 'pkey' in config:
                config['pkey'] = paramiko.RSAKey.from_private_key_file(
                    config['pkey']
                )

        # connect to  target machine through jump host
        jump.connect(**jump_config)
        tunnel = jump.get_transport().open_channel(
            "direct-tcpip",
            (target_config['hostname'], target_config['port']),
            (jump_config['hostname'], jump_config['port']),
        )
        target.connect(**target_config, sock=tunnel)

        # execute command
        stdin, stdout, stderr = target.exec_command(command)

        return stdout.read().strip().decode()
