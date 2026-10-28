import json
from pathlib import Path
import pytest
from openarmx_teleop_vr_306_v5.gripper_aperture import checked_closure, closure_to_motor, save_closure, load_closure


@pytest.mark.parametrize('width',[0,10,20,30,40,50,60,70,80,90,100])
def test_nominal_range(width):
    assert 10 <= closure_to_motor(width) <= 360


@pytest.mark.parametrize('width',[-1,100.1,float('nan'),float('inf'),True,None,'bad'])
def test_reject_invalid(width):
    with pytest.raises((TypeError,ValueError)):checked_closure(width)


def test_endpoints_and_monotonicity():
    assert closure_to_motor(0)==10
    assert closure_to_motor(100)==360
    assert all(closure_to_motor(i)<closure_to_motor(i+1) for i in range(100))


def test_saved_width_survives_reload(tmp_path):
    path=tmp_path/'settings.json'
    assert load_closure(path)==90
    save_closure(55,path)
    assert load_closure(path)==55
    with pytest.raises(ValueError):save_closure(101,path)
    assert load_closure(path)==55


def test_production_config_matches_recorded_pose():
    import yaml
    root=Path(__file__).resolve().parents[1]
    params=yaml.safe_load((root/'config/controller.yaml').read_text())
    p=next(iter(params.values()))['ros__parameters']
    assert p['quick_reset_left_arm_joints'][0]==pytest.approx(20.)
    assert p['quick_reset_right_arm_joints'][3]==pytest.approx(-110.)
    assert p['quick_reset_neck_joints'][1]==pytest.approx(0.)
    assert p['gripper_max_position']==360
    teleop=next(iter(yaml.safe_load((root/'config/teleop.yaml').read_text()).values()))['ros__parameters']
    assert 0 < teleop['gripper_filter_alpha'] <= 1
    assert 0 < teleop['gripper_max_step_per_cycle'] <= 350
