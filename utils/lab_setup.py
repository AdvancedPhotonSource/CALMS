import pandas as pd

import robotics as ro

# fmt: off
ro.runtime['rack_status'] = {
    # 0 = None (free_space), 1 = 'new', 2 = False (do_not_use)
    'vial': pd.DataFrame(
        [
            # 1st rack 6x8
            ['water_gap', 'NaCl', 'RFU_40mg/ml', 'RFU_50mg/ml', 'RFU_60mg/ml', 'RFU_100mg/ml', None, None],
            # ['water_gap', 'NaCl', 'RFU_30mg/ml', 'RFU_40mg/ml', 'RFU_50mg/ml', 'RFU_100mg/ml', None, None],
            # ['water_gap', 'NaCl', '30mg/ml', '40mg/ml', '50mg/ml', '60mg/ml', None, None],
            [False, None, 'PEDOT:PSS', None, None, None, None, None],
            [False, None, None, None, None, None, None, None],
            [None, None, False, None, None, None, None, None],
            [None, None, False, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            # 2nd rack 6x8
            [False, False, None, None, None, False, False, False],
            [False, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            [False, False, False, False, None, None, None, None],
        ]
    ),
    'pipette': pd.DataFrame(
        [
            # 4x24
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
    'substrate': pd.DataFrame(
        [
            # 12x6
            ['PDMS', 'PDMS', 'PDMS', 'PDMS', 'PDMS', 'PDMS'],
            ['250nm', '250nm', '250nm', '250nm', '250nm', '250nm'],
            ['500nm', '500nm', '500nm', '500nm', '500nm', '500nm'],
            ['1000nm', '1000nm', '1000nm', '1000nm', '1000nm', '1000nm'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
        ]
    ),
    'cooking': pd.DataFrame(
        [
            # 1x4
            ['40mg/ml', '50mg/ml', '60mg/ml', '100mg/ml'],
            # ['30mg/ml', '40mg/ml', '50mg/ml', '100mg/ml'],
            # ['40mg/ml', '60mg/ml', '80mg/ml', '100mg/ml'],
            # [None, None, None, None],
        ]
    ),
}
rack_status = ro.runtime['rack_status']

