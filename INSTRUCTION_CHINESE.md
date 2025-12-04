这是 `INSTRUCTION.md` 文件的中文翻译。为了保持技术术语的准确性，部分关键词（如 Kernel, Boid, Buffer, Thrust 等）保留了英文或进行了中英对照。

-----

# CUDA 入门 - 群聚模拟 (Flocking)

**截止日期：2025年9月7日，周日**

**摘要：** 在本项目中，你将获得编写简单 CUDA 核函数 (Kernel)、使用它们以及分析其性能的真实体验。你将基于 Reynolds Boids 算法实现一个群聚模拟，并进行两个层次的优化：均匀网格 (Uniform Grid) 和 具有半连续内存访问的均匀网格 (Uniform Grid with semi-coherent memory access)。

## Part 0: 准备工作 (Nothing New)

本项目（以及本课程中的所有其他 CUDA 项目）需要一张具备 CUDA 功能的 NVIDIA 显卡。任何计算能力 (Compute Capability) 在 3.0 (GTX 6xx 系列及更新) 或更高的显卡都可以使用。请在这个 [兼容性列表](https://developer.nvidia.com/cuda-gpus) 上检查你的 GPU。如果你个人没有具备这些规格的机器，你可以使用 Moore 100B/C 实验室中支持 GPU 的计算机，但请密切注意下方的 **设置 (setup)** 章节。

**注意**：如果你需要在实验室计算机上进行开发，你目前将无法进行 GPU 性能剖析 (Profiling)。这对于调试程序的性能瓶颈非常重要。如果你没有任何具有 CUDA 功能机器的管理员权限，请发邮件给 TA。

## Part 1: 朴素 Boids 模拟 (Naive Boids Simulation)

### 1.0. 设置 - 常规步骤

参考 Project 0 的 Part 1-3。

如果你使用的是 Nsight IDE（而不是 Visual Studio）并且提早开始了 Project 0，请注意情况略有变化。不要新建项目，而是使用 *File-\>Import-\>General-\>Existing Projects Into Workspace*，选择项目文件夹作为根目录。在 *Project-\>Build Configurations-\>Set Active...* 下，你现在可以选择各种 Release 和 Debug 构建版本。

  * `src/` 包含源代码。
  * `external/` 包含 GLEW, GLFW 和 GLM 的二进制文件和头文件。

**CMake 说明：** 不要直接在你的项目（Visual Studio, Nsight 等）中更改任何构建设置或添加任何文件。请编辑 `CMakeLists.txt` 文件。你创建的任何文件都必须添加在这里。如果你编辑了它，只需重新构建你的 VS/Nsight 项目以将更改同步到 IDE 中。

请在不修改代码的情况下运行项目，以确保一切工作正常。我们在 `kernel.cu` 中预留了运行测试代码的空间，目前它包含一个如何使用 `Thrust` 库在 GPU 上执行键值排序 (key-value sorting) 的示例。如果一切正常，你应该会在控制台窗口看到一些输出，并看到一个由灰色粒子组成的立方体。查看器配有相机控制：左键拖动移动视角，右键垂直拖动缩放。

**注意：请在 `release` 模式下构建项目以进行性能分析和捕捉。**

### 1.1. 具有朴素邻居搜索的 Boids

在 Boids 群聚模拟中，代表鸟或鱼（boids）的粒子根据三条规则在模拟空间中移动：

1.  **聚拢 (Cohesion)** - boids 移向其邻居的感知质心 (Center of Mass)。
2.  **分离 (Separation)** - boids 避免与其邻居靠得太近。
3.  **对齐 (Alignment)** - boids 通常试图模仿其邻居的方向和速度。

这三条规则规定了 boid 在一个时间步长内的速度变化。
在每个时间步长，一个 boid 必须检查它的每一个邻居 boid，并计算来自这三条规则的速度变化贡献。
因此，一个最基础的 boids 实现需要让每个 boid 检查模拟中的**每一个**其他 boid。

这是一些伪代码，可以帮助你：

#### 规则 1: Boids 试图飞向邻近 boids 的质心

```
function rule1(Boid boid)

    Vector perceived_center

    foreach Boid b:
        if b != boid and distance(b, boid) < rule1Distance then
            perceived_center += b.position
        endif
    end

    perceived_center /= number_of_neighbors

    return (perceived_center - boid.position) * rule1Scale
end
```

#### 规则 2: Boids 试图与其他物体（包括其他 boids）保持小段距离

```
function rule2(Boid boid)

    Vector c = 0

    foreach Boid b
        if b != boid and distance(b, boid) < rule2Distance then
            c -= (b.position - boid.position)
        endif
    end

    return c * rule2Scale
end
```

#### 规则 3: Boids 试图匹配附近 boids 的速度

```
function rule3(Boid boid)

    Vector perceived_velocity

    foreach Boid b
        if b != boid and distance(b, boid) < rule3Distance then
            perceived_velocity += b.velocity
        endif
    end

    perceived_velocity /= number_of_neighbors

    return perceived_velocity * rule3Scale
end
```

基于 [Conard Parker 的笔记](http://www.vergenet.net/~conrad/boids/pseudocode.html) 并稍作修改。为了使模拟更有趣，我们规定两个 boids 只有在彼此处于一定的 **邻域距离 (neighborhood distance)** 内时才会相互影响。

我们还有一个简单的 [Processing 上的 2D 实现](http://studio.sketchpad.cc/sp/pad/view/ro.9cbgCRcgbPOI6/rev.23)，它在概念上与你要编写的代码非常相似。你可以随意将此实现作为数学/代码参考。

为了了解 3D 模拟“应该”是什么样子，[这里是我们的参考实现的样子。](https://vimeo.com/181547860)

**请注意**，我们的伪代码、我们的 2D 实现以及我们的参考代码（我们从中得出了基础代码中附带的参数）在 **规则 3** 上与 Conrad Parker 的笔记不同——我们的参考没有从感知速度中减去 boid 自身的速度：

我们的伪代码：

```
    return perceived_velocity * rule3Scale
```

Conrad Parker 的笔记：

```
    RETURN (pvJ - bJ.velocity) / 8
```

这纯粹是因为制作参考代码的 TA 在最初创建这个项目时漏掉了这个小细节，但结果看起来还是挺对的 :facepalm:（捂脸）。

严格遵循 Conrad Parker 的伪代码可能会在使用我们项目附带的默认参数时导致意想不到的结果，你可能需要通过调整参数来解决。如果你发现了适用于“更正确”的 Boids 实现的好参数，欢迎在 Piazza 上分享！

然而，由于本作业的目的是介绍 CUDA，我们建议你目前在初步实现算法时遵循**我们的伪代码**，这样你就不必在调试实现的同时还要去调整参数。

### 1.2. 代码导读

  * `src/main.cpp`: 执行所有的 CUDA/OpenGL 设置和 OpenGL 可视化。
  * `src/kernel.cu`: CUDA 设备函数、状态、核函数 (kernels) 以及用于调用核函数的 CPU 函数。这里没有单元测试/沙盒框架，但预留了空间让你单独运行内核并在运行实际模拟之前从 GPU 获取输出。请在 **Part 2** 中利用这一点来单独测试你的内核。

<!-- end list -->

1.  在代码中搜索 `TODO-1.2` 和 `LOOK-1.2`。
      * `src/kernel.cu`: 利用你在第一节课中学到的知识，弄清楚如何解决 Part 1 的这些 TODO。

### 1.3. 尝试与探索

我们提供的参数在使用我们的参考实现时能产生稳定的模拟，但你的情况可能会有所不同。试着调整 boid 的数量，看看模拟如何响应。

## Part 2: 让群聚更上一层楼！ (Let there be (better) flocking\!)

### 2.0. 均匀网格 (Uniform Grids) 简述

回顾 Part 1，任意两个 boids 只有在彼此处于某个 *邻域距离* 内时才会相互影响。
基于这一观察，我们可以看到让每个 boid 检查每一个其他 boid 是非常低效的，特别是当 boid 数量很大且邻域距离远小于整个模拟空间时（如我们的标准参数所示）。我们可以使用一种称为 **均匀空间网格 (uniform spatial grid)** 的数据结构来剔除大量的邻居检查。

均匀网格由宽度至少为邻域距离的单元格组成，覆盖整个模拟域。
在计算 boids 的新速度之前，我们在预处理步骤中将它们“装箱 (bin)”到网格中。

如果单元格宽度是邻域距离的两倍，每个 boid 只需要检查 8 个单元格中的其他 boids，在 2D 情况下则是 4 个。

你可以通过遍历 boids，计算其所在的单元格，然后在一个代表单元格的可变长数组中保存指向该 boid 的指针，从而在 CPU 上构建均匀网格。然而，这不能很好地移植到 GPU 上，因为：

1.  我们在 GPU 上没有可变长数组。
2.  天真地并行化迭代可能会导致竞态条件 (race conditions)，即两个粒子需要在同一个时钟周期写入同一个桶。

相反，我们将通过**排序**来构建均匀网格。如果我们用代表其所在单元格的索引标记每个 boid，然后按这些索引对 boid 列表进行排序，我们可以确保同一单元格中的 boid 指针在内存中是连续的。

然后，我们可以遍历排序后的均匀网格索引数组，查看每对相邻的值。如果值不同，我们就知道我们处于两个不同单元格表示的边界。将这些位置存储在一个表中（每个单元格一个条目）就给了我们均匀网格的完整表示。这个“表”可以只是一个拥有与单元格数量相同空间的数组。这个过程是数据并行的，可以进行简单的并行化。

### 2.1. 代码导读

我们不会让你在第一次作业中实现 GPU 上的并行排序，而是使用内置于 **Thrust** 中的 value/key 排序。请参见 `kernel.cu` 中的 `Boids::unitTest` 以获取使用示例。

你的均匀网格在 GPU 内存中大概长这样：

  - `dev_particleArrayIndices` - 缓冲区，包含每个 boid 指向其在 `dev_pos`、`dev_vel1` 和 `dev_vel2` 中数据的指针。
  - `dev_particleGridIndices` - 缓冲区，包含每个 boid 的网格索引。
  - `dev_gridCellStartIndices` - 缓冲区，包含每个单元格指向其数据在 `dev_particleArrayIndices` 中起始位置的指针。
  - `dev_gridCellEndIndices` - 缓冲区，包含每个单元格指向其数据在 `dev_particleArrayIndices` 中结束位置的指针。

在这里，当用于缓冲区时，术语 `指针 (pointer)` 在很大程度上与 `索引 (index)` 可互换，但你实际上将使用数组索引作为指针。

查看代码中 Part 2.1 的 TODO 和 LOOK 以获得一些伪代码帮助。

你可以使用 `main.cpp` 中的宏定义在不同的时间步更新模式之间切换。

### 2.2. 进一步尝试

比较你的均匀网格速度更新与朴素速度更新。
在典型情况下，均匀网格版本应该快得多。
尝试推高你可以模拟的 boids 数量的极限。

将均匀网格的单元格宽度更改为邻域距离，而不是邻域距离的两倍。现在，需要检查 27 个相邻的单元格是否存在交集。这会增加还是减少群聚的效率？

### 2.3. 去除中间商 (Cutting out the middleman)

考虑 2.1 中概述的均匀网格邻居搜索：指向单个单元格中 boids 的指针在内存中是连续的，但 boid 数据本身（速度和位置）却分散在各处。尝试重新排列 boid 数据本身，使一个单元格中所有 boids 的速度和位置在内存中也是连续的，这样就可以直接使用 `dev_gridCellStartIndices` 和 `dev_gridCellEndIndices` 访问数据，而无需经过 `dev_particleArrayIndices`。

查看 Part 2.3 的 TODO。这应该涉及对你 2.1 代码的稍加修改的复制。

## Part 3: 性能分析 (Performance Analysis)

对于本项目，我们将通过一些基本问题指导你进行性能分析。将来，你将指导自己的性能分析——但回答这些简单的问题始终至关重要。总的来说，我们希望你能超越建议的性能调查，探索代码的不同方面如何影响整体性能。

提供的帧率计（在窗口标题中）将是一个有用的基础指标，但添加你自己的 `cudaTimer` 等将允许你对代码的各个部分进行更细粒度的基准测试。

记住：

  * 在 `Release` 模式下进行性能测试！
  * 在 Nvidia 控制面板中关闭垂直同步 (Vertical Sync):
  * 性能应始终尽可能相对于某个基线进行测量。GPU 可以让你的程序更快——但快了多少？
  * 如果某个更改影响了性能，请展示对比。描述你的更改。
  * 描述你用于基准测试的方法。
  * 性能图表是个好东西。

### 问题

有两种测量性能的方法：

  * 禁用可视化，这样报告的帧率将仅针对模拟本身，且不受限于 60 fps。这样，窗口标题中报告的帧率将很有用。
      * 要做到这一点，将 `#define VISUALIZE` 改为 `0`。
  * 为了更精确的时间测量，你可以使用 CUDA 事件 (CUDA events) 仅测量模拟的 CUDA 核函数。关于这方面的信息可以在网上轻松找到。你可能需要对几个模拟步长取平均值，类似于当前计算 FPS 的方式。

本部分不针对正确性评分，但请告诉我们你的假设和见解。

**回答这些问题：**

  * 对于每种实现，改变 boids 的数量如何影响性能？你认为这是为什么？
  * 对于每种实现，改变 block count (块数量) 和 block size (块大小) 如何影响性能？你认为这是为什么？
  * 对于连续均匀网格 (coherent uniform grid)：你是否体验到了比不连续版本更好的性能？这是否符合你的预期？为什么？
  * 改变单元格宽度并检查 27 个相邻单元格与 8 个相邻单元格相比，是否影响了性能？为什么？注意：仅仅因为要检查更多单元格就说 27 单元格版本更慢是不够的（而且可能是不正确的）！

**注意：Nsight 性能分析工具目前*无法*在实验室计算机上使用，因为它们需要管理员权限。** 如果你无法使用具有 CUDA 功能的计算机，实验室计算机仍然允许你进行计时测量！但是，这些工具对于性能调试非常有用。

## Part 4: 报告 (Write-up)

1.  截取 boids 的截图，**并且**使用像 [licecap](http://www.cockos.com/licecap/) 这样的 gif 工具录制一段固定视角的 boids 动画。
    把它放在你的 README.md 顶部。参考 [如何制作一个吸引人的 GitHub 仓库](https://github.com/pjcozzi/Articles/blob/master/CIS565/GitHubRepo/README.md)。
2.  添加你的性能分析。包括以下图表：
      - 帧率随 boids 数量增加的变化（针对朴素、离散均匀网格和连续均匀网格，分别在有和无可视化的情况下）。
      - 帧率随 block size (块大小) 增加的变化。

## 提交 (Submit)

如果你修改了任何 `CMakeLists.txt` 文件（除了 `SOURCE_FILES` 列表之外），请明确说明。注意 Google Group 上讨论的任何构建问题。

发起一个 GitHub Pull Request，以便我们可以看到你已完成。
标题应为 "Project 1: YOUR NAME"。
你的 Pull Request 评论部分的模板附在下面，你可以进行复制粘贴：

  * [Repo Link](https://www.google.com/search?q=https://link-to-your-repo)
  * (Briefly) Mentions features that you've completed. Especially those bells and whistles you want to highlight (简要提及你已完成的功能，特别是你想强调的那些亮点)
      * Feature 0
      * Feature 1
      * ...
  * Feedback on the project itself, if any. (对项目本身的反馈，如果有的话)

然后你就完成了！

## 提示 (Tips)

  - 如果你的模拟在启动前崩溃，请在 CUDA 调用后使用 `checkCUDAErrorWithLine("message")`。
  - Visual Studio 中的 `ctrl + f5` 将启动程序，但在程序崩溃时不会让窗口关闭。这样你可以看到任何 `checkCUDAErrorWithLine` 的输出。
  - 出于调试目的，你可以将数据传入传出 GPU。参见 `kernel.cu` 中的 `Boids::unitTest` 以获取如何使用的示例。
  - 对于像 4K 显示器或带 Retina 显示屏的 Macbook Pro 这样的高 DPI 显示器，你可能需要将渲染分辨率和点大小加倍。参见 `main.hpp`。
  - 你的 README.md 将使用 github markdown 编写。你可以在这里找到 [备忘单](https://guides.github.com/pdfs/markdown-cheatsheet-online.pdf)。github 还有用于 [atom 文本编辑器](https://atom.io/) 的 [实时预览插件](https://atom.io/packages/markdown-preview)。[VS Code](https://www.visualstudio.com/en-us/products/code-vs.aspx) 也是如此。
  - 如果你的帧率被限制在 60fps，请 [禁用 V-sync (垂直同步)](http://support.enmasse.com/tera/enable-v-sync-to-fix-graphics-issues-screen-tearing)。

## 可选的加分项 (Optional Extra Credit)

  * 共享内存优化 (Shared-Memory Optimization):
      * 使用共享内存和均匀网格添加快速最近邻搜索。包括额外的图表和性能分析，清楚地显示使用共享内存后程序性能提高了多少。
  * 网格循环优化 (Grid-Looping Optimization):
      * 不是硬编码搜索指定区域，而是根据具有最大距离 (max\_distance) 内的任何方面的网格单元来限制搜索区域。这防止了与每个网格单元角点的过多位置比较，同时也允许更灵活的方法（因为我们只是在所有三个基数方向上定义最小单元格索引和最大单元格索引）。也就是说，不再根据实现手动检查硬编码的特定数量的周围单元格（例如 8 个周围单元格，27 个周围单元格等）。
