"""Bounded, stationary reach capture from distinct coherent VR pose frames.

Network gaps restart a sample window, never relax motion freshness limits.
Only the caller persists a successful result; a failed attempt leaves the
previous calibration intact.
"""
import numpy as np


class ReachCalibration:
    MAX_AGE = .15
    WINDOW = .8
    MIN_SAMPLES = 20
    TIMEOUT = 15.

    def __init__(self, now):
        self.started = float(now)
        self.ready_at = now + 2.2
        self.deadline = now + self.TIMEOUT
        self.samples = []
        self.last_frame = None
        self.last_reason = '等待头显与双手的有效追踪'

    def update(self, now, frame_id, received_at, vectors, reason=''):
        if now >= self.deadline:
            return dict(state='failed', message='标定未完成：' + self.last_reason
                        + '；请保持佩戴头显、双手可见后重试', remaining=0.)
        if now < self.ready_at:
            return dict(state='countdown', message='双臂自然向前伸直，松开Grip，保持头部稳定',
                        remaining=round(self.ready_at - now, 1))
        age = now - received_at if received_at is not None else float('inf')
        if not reason and not 0. <= age <= self.MAX_AGE:
            reason = '追踪帧暂时中断，正在等待新帧'
        if reason or vectors is None:
            self.samples.clear()
            self.last_reason = reason or '等待头显与双手的有效追踪'
            return dict(state='waiting_tracking', message=self.last_reason,
                        remaining=round(self.deadline-now, 1), samples=0)
        # Timer ticks and button events must never count as new pose samples.
        if frame_id != self.last_frame:
            self.last_frame = frame_id
            point = np.asarray(vectors, dtype=float)
            if point.shape != (2, 3) or not np.all(np.isfinite(point)):
                self.samples.clear()
                self.last_reason = '追踪坐标无效'
                return dict(state='waiting_tracking', message=self.last_reason, samples=0)
            if self.samples and (received_at <= self.samples[-1][0]
                                 or received_at-self.samples[-1][0] > self.MAX_AGE):
                self.samples.clear()
            self.samples.append((received_at, point.copy()))
            points = np.asarray([p for _, p in self.samples])
            if np.max(np.linalg.norm(points-points.mean(axis=0), axis=2)) > .03:
                self.samples = [(received_at, point.copy())]
                self.last_reason = '姿态尚未稳定，请保持头部和双臂不动'
                return dict(state='sampling', message=self.last_reason, progress=0., samples=1)
        span = self.samples[-1][0]-self.samples[0][0] if self.samples else 0.
        if span >= self.WINDOW and len(self.samples) >= self.MIN_SAMPLES:
            mean = np.mean([p for _, p in self.samples], axis=0)
            lengths = np.linalg.norm(mean, axis=1)
            if not np.all((lengths > .25) & (lengths < .95)) or np.any(mean[:, 0] < .15):
                self.samples.clear()
                self.last_reason = '请双臂自然向前伸直，并让头显看清手柄'
                return dict(state='sampling', message=self.last_reason, progress=0., samples=0)
            return dict(state='complete', lengths={s:float(lengths[i]) for i,s in enumerate(('left','right'))})
        self.last_reason = '有效样本不足，请保持头显与双手持续追踪、姿态稳定'
        return dict(state='sampling', message='正在采集稳定姿态，请保持不动',
                    progress=round(min(span/self.WINDOW, len(self.samples)/self.MIN_SAMPLES), 2),
                    samples=len(self.samples))
