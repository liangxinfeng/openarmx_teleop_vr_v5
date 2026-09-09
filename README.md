# 云蝶 WebXR 双臂遥操 V4

**307机器人当前版本快照，2026-09-09。** 包名沿用 `openarmx_teleop_vr_306_v4`，实际默认话题后缀为 `0_307`，不是306专用配置。

## 技术链路

```text
Quest浏览器手柄 → HTTPS / WebRTC / WSS最新帧
→ 相邻帧末端增量映射 → Placo QP IK与奇异点相关约束
→ 关节约束、过期结果检查、掉线保持、控制权管理
→ 厂商SYNC位置指令 → 厂商CAN层 → DM电机内部位置闭环
```

本包提供双臂、夹爪、头部/腰部选项、复位以及RGB视频。**厂商硬件服务和电机位置内环不在本仓库中**；与导航组合使用时，由导航包提供网页及额外按键逻辑。

## 环境与编译

需要Ubuntu 24.04、ROS 2 Jazzy、厂商 `robot_env` Python环境及硬件SDK。Placo使用独立虚拟环境；相机、WebRTC、轨迹生成等依赖还需与原机环境一致，rosdep不能安装厂商SDK。

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/liangxinfeng/openarmx_teleop_vr_306_v4.git
cd openarmx_teleop_vr_306_v4
bash scripts/setup_v4_placo_env.sh
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src/openarmx_teleop_vr_306_v4 --ignore-src -r -y
colcon build --packages-select openarmx_teleop_vr_306_v4
source install/setup.bash
ros2 launch openarmx_teleop_vr_306_v4 full_vr_teleop.launch.py \
  topic_suffix:=0_307 dry_run:=true
```

头显访问 `https://<机器人IP>:8446`。先在dry-run检查输入、映射和IK；确认编号、零位、限位、急停和现场安全后，才将 `dry_run:=false`，并在网页主动使能。

## 配置与迁移

| 内容 | 位置 |
| --- | --- |
| 关节、安全和控制参数 | `config/controller.yaml` |
| 手柄映射与跟随参数 | `config/teleop.yaml` |
| 启动默认值、端口和DDS | `launch/full_vr_teleop.launch.py` |
| 机器人模型 | `urdf/`、`meshes/` |
| Placo环境 | `scripts/setup_v4_placo_env.sh` |
| 回归测试 | `test/`、`integration/` |

迁移时不能只改机器人编号：核对模型、关节顺序/方向/限位、厂商服务协议和环境路径。当前监督遥操配置中的自碰撞开关应逐项检查，不能假设所有保护默认启用。

不要同时开启V3、V4和导航遥操抢占硬件；导航包已经包含本包，无需另开一套。测试脚本可能需要ROS/SDK，有些会发布控制消息，不应在真机域盲目执行。

## 相关项目与许可

- [VR导航与任务编排](https://github.com/liangxinfeng/openarmx_teleop_vr_navigation_306)
- [迁移指南](AI_MIGRATION_GUIDE.md)
- [原始算法说明](docs/ORIGINAL_README.md)
- [第三方许可证](THIRD_PARTY_LICENSES.md)、[NOTICE](NOTICE)

保留Quest3-Teleoperation等原有声明，厂商模型和硬件SDK权利不因公开托管而改变。本次为源代码发布，不代表进行了新的真机安全验收。
