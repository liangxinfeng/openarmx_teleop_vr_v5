# OpenArmX VR 遥操作 V5

本仓库的 `v5` 分支保存 345 机器人上 V5 遥操测试功能包的源码快照。历史 V4 代码保留在 `v4` 分支；原有 `main` 与机器人版本分支的提交历史也保留。

为保持与现有 ROS 2 启动文件和依赖兼容，V5 快照中的 ROS 功能包和 Python 模块仍命名为 `openarmx_teleop_vr_306_v5`。GitHub 仓库名不等同于 ROS 包名。

## 功能与实现

- WebXR 头显与手柄输入，经 WebSocket / WebRTC 传输至 ROS 2；支持双臂 Grip 接管、夹爪及头部跟随。
- 双臂七自由度遥操作使用 Mink 微分逆运动学和 quadprog；MuJoCo 用于运动学模型，不运行物理仿真。具体任务和约束见 [V5_MINK.md](V5_MINK.md)。
- 支持按 Grip 会话锁定臂形参考、反馈约束、失联保持及 X+A 复位路径检查。腰部前倾辅助的参数位于 `config/controller.yaml`。
- 本仓库是独立 V5 遥操测试包；345 的 VR 遥操导航界面、航点、地图和导航专用启动参数属于另一个功能包，不包含在此快照中。

## 依赖和启动

本包依赖 ROS 2、机器人原厂服务、机器人模型与 `control_center_joint_control`。它使用机器人环境中已有的 Pinocchio 等本地依赖；Mink 环境版本见 `requirements-mink.txt` 和 `scripts/setup_v5_mink_env.sh`。迁移至其他机器时，请先核对型号、关节限位、SDK、话题后缀与各脚本的本地路径。

在 ROS 2 工作空间内安装依赖并编译后，可先使用模拟模式检查启动：

```bash
cd ~/ros2_ws
colcon build --packages-select openarmx_teleop_vr_306_v5 --symlink-install
source install/setup.bash
ros2 launch openarmx_teleop_vr_306_v5 full_vr_teleop.launch.py dry_run:=true
```

启动文件默认网页端口为 `8447`。真机运行需要对应机器人配置、原厂服务以及现场验证；模拟启动不代表真机可安全运行。

## 快照说明

代码取自 345 机器人 `ros2_ws/src/openarmx_teleop_vr_306_v5`，采集日期 2026-10-28。仓库文档更新了本首页；源功能包没有在机器人上改名。`README_HISTORICAL_SNAPSHOT.md`、`V5_VALIDATION.json` 等历史文档保留原样，其旧版本算法描述或测试结果不应视为本次快照的最新验收结论。

来源和第三方许可见 [NOTICE](NOTICE) 与 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。
