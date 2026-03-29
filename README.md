# CEH-Flow-Perception
# CEH-流体驱动的心智

A real-time dual-field spatiotemporal perception and physical avoidance engine.

一个实时双场时空感知与物理避障引擎。

## 1. What This Project Is
## 1. 这个项目是做什么的

Chen-Flow is a perception-driven physical field engine for dynamic obstacle awareness, spatiotemporal conductance memory, agent back-reaction, and emergent avoidance behavior.

Chen-Flow 是一个由感知驱动的物理场引擎，用于动态障碍感知、时空导通记忆、智能体反作用以及涌现式避障行为。

Instead of relying only on explicit object boxes and a conventional planner, this project models dynamic obstacles as field sources and lets agents move under field forces.

本项目不是只依赖显式检测框和传统规划器，而是将动态障碍建模为场源，并让智能体在场力作用下运动。

The goal is not merely detection, but a continuous physical interpretation of risk in space and time.

目标不只是检测，而是在空间与时间上连续地解释风险。

## 2. Core Idea
## 2. 核心思想

The engine maintains at least two interacting fields:

引擎至少维护两个相互作用的场：

- a fast conduction field
- 一个快速导通场

- a slower learning or memory field
- 一个较慢的学习或记忆场

A simplified form is:
其简化形式为：

$$
C_t = \rho C_{t-1} - \gamma + \alpha M_t + \lambda L_t
$$

$$
L_t = (1-\eta)L_{t-1} + \eta M_t
$$

Here, $M_t$ is motion-derived input, $C_t$ is the fast field, and $L_t$ is the slower memory field.

其中，$M_t$ 是由运动得到的输入，$C_t$ 是快速场，$L_t$ 是较慢的记忆场。

The local force applied to an agent is computed from the potential gradient:

作用于智能体的局部力由势场梯度计算得到：

$$
\vec{F} = - \nabla V
$$

The important point is not the isolated equation itself, but the concrete implementation of dual-field coupling, normalization, back-reaction, adaptive tuning, and audit logic in this repository.

重要的不是孤立公式本身，而是本仓库中双场耦合、归一化、反作用、自适应调参和审计逻辑的具体实现。

## 3. Key Features
## 3. 主要特性

- real-time camera-based spatiotemporal field generation
- 基于摄像头的实时时空场生成

- conduction memory trail rendering
- 导通记忆轨迹渲染

- field-driven agent movement
- 场驱动智能体运动

- swarm dynamics and social force interaction
- 集群动力学与社会力交互

- agent back-reaction to the field
- 智能体对场的反作用

- risk gradient, repulsion force, and collision entropy audit
- 风险梯度、排斥力和碰撞熵审计

- human-readable Chinese UI and professional terminal debug logs
- 面向普通用户的中文界面与面向专业人员的终端调试日志

## 4. Why It Is Different
## 4. 它为什么不同

Traditional pipelines are often explained as:

传统流程通常可概括为：

$$
\text{Perception} \rightarrow \text{Prediction} \rightarrow \text{Planning} \rightarrow \text{Control}
$$

Chen-Flow emphasizes:

Chen-Flow 更强调：

$$
\text{Perception} \rightarrow \text{Field} \rightarrow \text{Force} \rightarrow \text{Motion}
$$

This project explores whether obstacle response can emerge from field dynamics rather than only from explicit symbolic planning logic.

本项目探索的是，障碍响应是否可以更多地从场动力学中涌现出来，而不是只依赖显式符号规划逻辑。

## 5. Performance Positioning
## 5. 性能定位

This repository is designed for real-time interactive demonstrations and engineering experiments.

本仓库主要面向实时交互演示与工程实验。

Typical goals include:

典型目标包括：

- low-latency field update
- 低延迟场更新

- real-time camera feedback
- 实时摄像头反馈

- high-visibility audit panels
- 高可见度审计面板

- scalable swarm behavior experiments
- 可扩展的集群行为实验

Exact performance depends on hardware, camera resolution, grid resolution, and enabled features.

具体性能取决于硬件、摄像头分辨率、网格分辨率与启用特性。

## 6. Commercial Use Notice
## 6. 商业使用提示

This repository is publicly available for research, learning, and open-source use under the applicable license.

本仓库公开提供，用于研究、学习以及在适用许可下的开源使用。

If you need closed-source commercial integration, paid deployment, SaaS use, internal proprietary deployment, OEM integration, or other commercial use without AGPLv3 compliance, please request a separate commercial license.

如果你需要闭源商业集成、收费部署、SaaS 使用、专有内部部署、OEM 集成或其他不遵守 AGPLv3 的商业使用，请申请单独的商业授权。

See:

请参见：

- LICENSE.md
- LICENSE.md

- COMMERCIAL_LICENSE.md
- COMMERCIAL_LICENSE.md

## 7.  Build and Run (Mac)
## 7.  编译与运行 (Mac)

A typical Qt and OpenCV workflow may look like this:

一个典型的 Qt 与 OpenCV 工作流可能如下：

```bash
git clone https://github.com/chenenhua/CEH-Flow-Perception
cd CEH-Flow-Perception
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Your exact build commands may differ depending on your Qt version, OpenCV installation, operating system, and compiler environment.

具体编译命令会因 Qt 版本、OpenCV 安装方式、操作系统和编译器环境而有所不同。

## 8. Intended Users

## 8. 适用人群

This project is suitable for:

本项目适合以下人群：
- robotics researchers
- 机器人研究人员
- autonomous system engineers
- 自动系统工程师
- industrial vision developers
- 工业视觉开发者
- intelligent manufacturing teams
- 智能制造团队
- computer vision experimenters
- 计算机视觉实验者
- people interested in field-based navigation and emergent control
- 对基于场的导航和涌现控制感兴趣的人

## 9. Legal and IP Clarification

## 9. 法律与知识产权澄清

This repository protects the specific implementation, code, documentation, UI expression, and other original expressive materials in this project.

本仓库保护本项目中的具体实现、代码、文档、界面表达以及其他原创表达材料。

This repository does not claim ownership over general physical laws, mathematical formulas, or independently developed implementations that do not copy protected expression from this project.

本仓库不主张对一般物理定律、数学公式或未复制本项目受保护表达内容的独立实现享有所有权。

## 10. Contact

## 10. 联系方式

陈恩华

15557000007

a106079595@qq.com
