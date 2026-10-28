"""345 DM Type1 logical travel; percent is travel, not physical aperture."""
import json
import math
import os
from pathlib import Path

STATE_PATH = Path.home() / '.config/openarmx_vr/gripper_aperture.json'
DM_STATE_PATH = Path.home() / '.config/openarmx_vr/gripper_345_dm_type1.json'


def checked_closure(value):
    if isinstance(value, bool):
        raise ValueError('DM夹爪闭合行程应为0–100%')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError('DM夹爪闭合行程应为0–100%')
    return value


def closure_to_motor(value):
    # Native driver maps logical 10..360 to physical motor 7..107 degrees.
    return 10.0 + checked_closure(value) * 3.5


def load_closure(path=DM_STATE_PATH):
    if not path.exists():
        return 90.0
    data = json.loads(path.read_text())
    if data.get('robot_id') != 345 or data.get('hardware') != 'dm_type1':
        raise ValueError('夹爪设置不是345 DM Type1配置')
    return checked_closure(data['closed_percent'])


def save_closure(value, path=DM_STATE_PATH):
    value = checked_closure(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'robot_id': 345, 'hardware': 'dm_type1', 'closed_percent': value}))
    os.replace(tmp, path)


def checked_width(value):
    if isinstance(value, bool):
        raise ValueError('闭合宽度必须是0–9cm的数值')
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 9.0:
        raise ValueError('闭合宽度必须在0–9cm之间')
    return value


def width_to_motor(value):
    raise ValueError('345 DM夹爪尚无毫米开度标定，请使用闭合行程百分比')


def load_width(path=STATE_PATH):
    if not path.exists():
        return 0.0
    return checked_width(json.loads(path.read_text())['closed_width_cm'])


def save_width(value, path=STATE_PATH):
    value = checked_width(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'closed_width_cm': value, 'mapping': 'nominal_95mm_v1'}))
    os.replace(tmp, path)
