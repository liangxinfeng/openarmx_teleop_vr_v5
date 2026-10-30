# OpenArmX VR 遥操作 V5 — 345 当前源码快照

本分支 `robot345-v5-20261030` 保存 2026-10-30 从 345 机器人读取的当前 V5 功能包。它在现有 `v5` 提交历史上新增；`v5`、`v4`、`main` 及其他原有分支不变。

ROS 功能包和 Python 模块名仍为 `openarmx_teleop_vr_306_v5`，保持现有启动文件、ROS 话题和上层依赖兼容。GitHub 仓库名为 `openarmx_teleop_vr_v5`。

## 本次快照

- 包含当前 Grip 接管与目标映射、手臂跟随、近伸直姿态约束、腰部补偿及 X+A 复位相关实现。
- 当前源码配置的腰部前倾补偿上限为 12°，包含重复接管时的腰部参考处理；实际启动入口仍可能覆盖参数。
- 保留原有代码、配置、模型、测试、网页和第三方声明，未为上传修改机器人算法或运行参数。
- 具体文件差异、来源和验证范围见 [本次发布说明](SNAPSHOT_345_20261030.md)。

## 依赖与范围

需要 ROS 2、机器人原厂服务、`control_center_joint_control` 及相应机器人模型和运动学环境。Mink/quadprog 等依赖见 `requirements-mink.txt` 与 `scripts/setup_v5_mink_env.sh`。MuJoCo 在本包中用于相关运动学求解，不代表运行整机物理仿真。

这是独立 V5 遥操功能包，不是完整机器人系统镜像。VR 遥操导航上层界面、导航地图、抓取与动作编排器、用户录制轨迹、机外环境以及原厂服务配置不在此快照中。345 上层 VR 遥操导航可以依赖本包，但须另外部署其自身功能包。

迁移前核对原厂 SDK、关节定义、机器人型号、话题后缀及脚本路径。安装依赖并编译后，可先运行无动作模式：

```bash
cd ~/ros2_ws
colcon build --packages-select openarmx_teleop_vr_306_v5 --symlink-install
source install/setup.bash
ros2 launch openarmx_teleop_vr_306_v5 full_vr_teleop.launch.py dry_run:=true
```

默认网页端口为 8447；模拟启动检查不等同于真机动作验收。

## 历史文档与来源

机器人原有首页保存在 [README_ROBOT_SNAPSHOT.md](README_ROBOT_SNAPSHOT.md)。`V5_CHANGES.md`、`V5_NATURAL_REACH.md`、`V5_VALIDATION.json` 等为不同时间的历史说明，可能包含旧算法描述或旧测试结果；它们不代表本次重新完成了全部功能验收。

本次源码校验清单见 [SOURCE_SNAPSHOT_345_20261030.json](SOURCE_SNAPSHOT_345_20261030.json)。第三方来源和许可见 [NOTICE](NOTICE) 与 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。
