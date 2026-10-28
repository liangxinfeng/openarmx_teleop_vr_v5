import numpy as np
from ._subject import load_subject

Capture = load_subject('reach_calibration').ReachCalibration
V = np.array([[.62, 0., 0.], [.64, 0., 0.]])


def fill(c, start, n=60, dt=.02):
    for i in range(n):
        t=start+i*dt
        result=c.update(t, t, t, V)
        if result['state']=='complete':
            return result
    return result


def test_distinct_stable_frames_complete_after_time_coverage():
    c=Capture(0.)
    assert c.update(1., 1, 1., V)['state']=='countdown'
    r=fill(c, 2.21)
    assert r['state']=='complete'
    np.testing.assert_allclose([r['lengths']['left'], r['lengths']['right']], [.62,.64])


def test_transient_gap_restarts_then_recovers_not_fails():
    c=Capture(0.)
    fill(c, 2.21, n=15)
    assert c.update(2.9, 2.49, 2.49, V)['state']=='waiting_tracking'
    assert len(c.samples)==0
    assert fill(c, 3.)['state']=='complete'


def test_gap_between_arrivals_resets_even_without_intermediate_tick():
    c=Capture(0.)
    fill(c, 2.21, n=15)
    r=c.update(3., 50, 3., V)
    assert r['samples']==1


def test_repeated_ticks_do_not_count_as_poses():
    c=Capture(0.)
    for i in range(100):
        r=c.update(2.21+i*.001, 1, 2.21, V)
    assert len(c.samples)==1
    assert r['state']!='complete'


def test_missing_head_reports_device_and_times_out():
    c=Capture(0.)
    r=c.update(2.3, 1, 2.3, None, '等待头显有效追踪')
    assert r['state']=='waiting_tracking'
    assert '头显' in r['message']
    assert c.update(15.1, 1, 2.3, None)['state']=='failed'


def test_motion_restarts_window_and_can_recover():
    c=Capture(0.)
    fill(c, 2.21, n=15)
    r=c.update(2.52, 90, 2.52, V+.1)
    assert r['samples']==1
    assert fill(c, 3.)['state']=='complete'


def test_implausible_arm_lengths_not_accepted():
    c=Capture(0.)
    for i in range(80):
        t=2.21+i*.02
        assert c.update(t,i,t,V*3)['state']!='complete'


def test_invalid_numeric_pose_not_accepted():
    c=Capture(0.)
    assert c.update(2.3, 1, 2.3, V*np.nan)['state']=='waiting_tracking'


def test_sparse_frames_need_twenty_distinct_samples():
    c=Capture(0.)
    r=fill(c, 2.21, n=10, dt=.1)
    assert r['state']=='sampling'
    assert r['samples']==10


def test_future_timestamp_cannot_count_as_fresh():
    c=Capture(0.)
    assert c.update(2.3, 1, 3., V)['state']=='waiting_tracking'
