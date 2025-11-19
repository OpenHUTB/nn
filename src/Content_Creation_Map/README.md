# RoadRunner 使用教程说明

关于 RoadRunner 的使用教程，教师提供的 RoadRunner R2022b 版本并没有汉化，在网上很难找到系统的 RoadRunner 教学视频。

附上 MathWorks 的官方使用文档网址：  
https://ww2.mathworks.cn/help/roadrunner/index.html?s_tid=CRUX_lftnav

---

## 注册 MathWorks 账号注意事项

当你没有 MathWorks 的账号时，需要进行注册。以下是注册过程中需要注意的事项：

### 1. 检查网络连接

- **使用稳定网络**：避免使用不稳定的公共 Wi-Fi。如果可能，请切换到有线网络或手机热点（4G/5G）再试。
- **关闭 VPN/代理**：某些 VPN 或网络代理服务器可能会被 MathWorks 的安全机制阻止。请务必先关闭它们，再尝试注册。
- **清除浏览器缓存**：有时旧的缓存或 Cookie 会导致问题。请清除浏览器缓存和 Cookie，或尝试使用浏览器的“无痕/隐私模式”进行注册。

### 2. 验证邮箱地址

- **邮箱是否有效**：确保您使用的邮箱地址（推荐学校邮箱或公司邮箱，其次是 Outlook/Gmail 等国际邮箱）是正确的且可以正常接收邮件。
- **是否已注册**：如果您之前用这个邮箱注册过，系统会提示“此电子邮件地址已与某个帐户关联”。请尝试使用“忘记密码”功能来重置密码，而不是重新注册。
- **避免使用某些国内邮箱**：例如 QQ 邮箱、163 邮箱等，有时可能会因为邮件过滤或服务问题导致收不到验证邮件。如果遇到问题，建议换用 Gmail 或 Outlook 邮箱。

### 3. 仔细检查注册信息（非常重要！）

注册信息，尤其是姓名和出生日期，必须真实有效，这在后续的学术验证或许可证激活中至关重要。

- **姓名**：请使用英文（拼音）按照“名 First Name”和“姓 Last Name”的顺序填写。例如，张三应填写为：
  - First Name: San
  - Last Name: Zhang
- **出生日期**：请确保填写正确。
- **国家/地区**：选择您所在的国家/地区。
- **行业/角色**：根据您的实际情况选择，例如 Student（学生）或 Academic Researcher（学术研究人员）。

### 4. 满足密码要求

MathWorks 的密码要求比较严格，请确保您的密码包含：

- 至少 8 个字符。
- 至少一个大写字母（A-Z）。
- 至少一个小写字母（a-z）。
- 至少一个数字（0-9）。
- 不能包含您的用户名（邮箱）中的连续三个字母。


# RoadRunner的下载

通过网盘分享的文件：RoadRunner_2022b.zip
链接: https://pan.baidu.com/s/1fA7jpkxtMA5uooNscgkQqA?pwd=1aaa 提取码: 1aaa


# 一个简单的roadrunner场景
<<<<<<< HEAD

通过网盘分享的文件：1.zip
链接: https://pan.baidu.com/s/1JmsUv6jfPXRF-SCJmUjzkg?pwd=1aaa 提取码: 1aaa

## 使用方法
- **将文件解压缩到本地**：确保完整解压
- **找到文件地址**："1/Scenes/1.rrscene"，右键选择打开方式为roadrunner

# 新建一个简单的roadrunner场景流程


![A simple RoadRunner scene](https://ww2.mathworks.cn/help/roadrunner/ug/gs_final_scene.png)

### 前提条件

在开始此示例之前，请确保您的系统满足以下前提条件：

-   您已按照[安装并激活 RoadRunner](https://ww2.mathworks.cn/help/roadrunner/ug/install-and-activate-roadrunner.html)中描述的说明下载、安装并激活 RoadRunner。
    
-   您拥有 [RoadRunner Asset Library 附加组件](https://ww2.mathworks.cn/help/roadrunner/roadrunner-asset-library.html) 的许可证。此示例使用仅在该库中可用的素材。
    

虽然此示例涵盖了一些基本的相机操作，但为了更全面地了解 RoadRunner 相机的工作原理，请先查看 [RoadRunner 中的相机控制](https://ww2.mathworks.cn/help/roadrunner/ug/camera-control-in-roadrunner.html) 示例。

### 创建新场景和工程

在 RoadRunner 中，您创建的每个场景都是_工程_的一部分，该工程是一个素材（场景组件）文件夹，可以在该工程的所有场景之间共享。创建一个新场景和一个放置该场景的新工程。

1.  打开 RoadRunner，然后从开始页面点击 **New Scene**。
    
2.  在选择工程窗口中，点击 **New Project**。
    
3.  在文件系统中，浏览到要在其中创建工程的空文件夹。如果不存在空文件夹，请创建一个并将其命名 `My Project`。文件夹名称将成为工程的名称。
    
4.  出现提示时，点击 **Yes** 在您的工程中安装 RoadRunner Asset Library。
    

RoadRunner 打开一个新场景，其中有一个空白的场景编辑画布。

![Empty RoadRunner scene editing canvas](https://ww2.mathworks.cn/help/roadrunner/ug/gs_empty_canvas.png)

您指定的工程名称出现在标题栏中。场景的名称也会出现在标题栏中，但在您保存场景并命名之前，它会显示为 **New Scene**。

![RoadRunner title bar](https://ww2.mathworks.cn/help/roadrunner/ug/gs_title_bar.png)

您可以随时从 菜单创建新场景、更改场景或更改工程。当您重新打开 RoadRunner 时，您可以在开始页的**最近的场景**列表中选择您最近处理的场景。

### 添加道路

当您打开一个新场景时，RoadRunner 会打开并选定 **Road Plan Tool** ![](https://ww2.mathworks.cn/help/roadrunner/ug/icon_tool_road_plan56d7dcdfe4d48f4ef820074f50f5b58d.png)。有关使用此工具的说明显示在底部状态栏中。通过在选择此工具的情况下在场景编辑画布中右键点击，您可以添加塑造道路几何形状的控制点。

1.  在场景编辑画布的底部中心，右键点击以添加新道路的第一个控制点。
    
    ![Red control point at bottom-center of canvas](https://ww2.mathworks.cn/help/roadrunner/ug/gs_control_point1.png)
    
2.  在画布的顶部中心，右键点击以添加第二个控制点并形成第一个路段。
    
    ![Road segment running from bottom to top of canvas](https://ww2.mathworks.cn/help/roadrunner/ug/gs_control_point2.png)
    
3.  在远离道路的地方点击以取消选择道路并完成创建。
    
    ![Finished road, no longer selected](https://ww2.mathworks.cn/help/roadrunner/ug/gs_commit_road.png)
    
4.  通过右键点击第一条道路的左侧、右键点击其右侧，然后点击远离道路的位置，创建一条与第一条道路相交的新直线道路。两条路形成一个交叉口。
    
    ![Straight intersecting roads that form a junction](https://ww2.mathworks.cn/help/roadrunner/ug/gs_intersection.png)
    

到目前为止，您已经创建了笔直的道路。要形成弯曲道路，请右键点击多次以向道路添加其他控制点。创建一条与交叉路口重叠的弯曲道路。

1.  在交叉路口的左上象限内点击鼠标右键。
    
2.  在交叉路口的右上象限内右键点击。第一个创建的路段是直的。
    
3.  右键点击交叉路口的右下象限。交叉路口和弯曲道路围成的区域形成地面。
    
    ![Curved road added to intersection in three steps](https://ww2.mathworks.cn/help/roadrunner/ug/gs_curve_montage.png)
    

您可以通过选择道路端点并右键点击添加更多控制点来延伸现有道路。

1.  在您创建的弯曲道路中，点击以选择画布顶部附近的末端。
    
2.  右键点击交叉路口的左端。RoadRunner 创建一条满足必要几何约束的道路。封闭区域再次形成地面。
    
    ![Road connecting the left side of the curved road to the left side of the intersection](https://ww2.mathworks.cn/help/roadrunner/ug/gs_road_end_montage.png)
    

要修改任何道路，请点击将其选中，然后尝试拖动其控制点或移动整条道路。您还可以右键点击道路来添加其他控制点。例如，在此道路网络中，您可以添加控制点来平滑交叉路口左侧的曲线。

### 添加表面地形

到目前为止，只有道路包围的区域包含地表地形。要在整个道路网络周围添加表面地形，您可以使用表面工具 ![](https://ww2.mathworks.cn/help/roadrunner/ug/icon_tool_surface92e21d640f134e753ec622b0611b2d08.png)。

1.  在工具栏中，点击 按钮。选择一个新工具将使 RoadRunner 处于不同的模式，从而实现新的交互并使不同的场景对象可供选择。选择 后，道路不再可选，但路面节点变为可选。
    
    ![Road networks with surface nodes selectable](https://ww2.mathworks.cn/help/roadrunner/ug/gs_surface_tool_selected.png)
    
2.  缩小场景，可以使用滚轮或按住 **Alt** 并右键点击，然后向下或向左拖动。
    
    ![Road network zoomed out](https://ww2.mathworks.cn/help/roadrunner/ug/gs_surface_zoom_out.png)
    
3.  右键点击道路网络上方以添加新的表面节点。然后，继续右键点击道路周围的点以形成一个圆圈。当您再次到达顶部节点时，右键点击它以连接曲面图并将曲面提交到画布。
    
    ![Surface being added around road network in 6 steps](https://ww2.mathworks.cn/help/roadrunner/ug/gs_surface_montage.png)
    

要修改曲面尺寸，请点击并拖动曲面节点。要修改曲面的曲线，请点击节点之间的线段，然后点击并拖动切线。

### 添加高程和桥梁

至此，场面已经平淡。通过更改其中一条道路的高度来修改场景中的高程。

1.  按住 **Alt**，然后点击并拖动相机以一定角度查看场景。
    
    ![Scene viewed at an angle](https://ww2.mathworks.cn/help/roadrunner/ug/gs_roads_at_angle.png)
    
2.  点击 **Road Plan Tool** 按钮可再次选择道路。然后，点击以选择您创建的第一条弯曲道路。
    
    ![Curved road selected](https://ww2.mathworks.cn/help/roadrunner/ug/gs_select_road_to_elevate.png)
    
3.  要升高道路，请使用 **2D Editor**，它可以让您查看道路轮廓和道路横截面等场景方面。在 **2D Editor** 中，选择道路的轮廓并将其提高约 10 米。
    
    ![On left, 2D Editor with road flat. On right, 2D Editor with road raised 10 meters.](https://ww2.mathworks.cn/help/roadrunner/ug/gs_2d_editor_montage.png)
    
    现在，道路在场景画布中的交叉路口上方已升高。高架道路不是形成交叉口，而是形成立交桥。
    
    ![Curved road elevated above the intersection](https://ww2.mathworks.cn/help/roadrunner/ug/gs_elevate_road.png)
    

道路依附于地表地形。当您抬高道路时，地形也会随之抬高。增加高程可能会导致立交桥下方出现视觉伪影。为了解决此问题，您可以使用道路构造工具 ![](https://ww2.mathworks.cn/help/roadrunner/ug/icon_tool_road_constructione21bc0a8e0054a3469c85d27dccb7d03.png) 创建桥梁跨度。

1.  旋转相机并放大以查看立交桥上的视觉伪影。
    
    ![Road with visual artifacts present](https://ww2.mathworks.cn/help/roadrunner/ug/gs_road_visual_artifacts.png)
    
2.  点击 Road Construction Tool 按钮。
    
3.  在左侧工具栏上，点击 **Auto Assign Bridges** 按钮 ![](https://ww2.mathworks.cn/help/roadrunner/ug/icon_auto_assign_bridges.png)。此操作仅在使用道路构造工具时可用，它仅将位于区域正上方的道路部分转换为桥梁跨度。使用默认的桥梁跨度膨胀并点击 **OK**。道路跨度被转换为桥梁，视觉伪影被消除。
    
    ![Road with bridge spans and no visual artifacts](https://ww2.mathworks.cn/help/roadrunner/ug/gs_road_with_bridges.png)
    
    如果桥梁形成不正确，请尝试调整道路高程或桥梁跨度膨胀并重新运行 **Auto Assign Bridges** 操作。
    

### 修改交叉口

某些工具使您能够选择和修改交叉口的属性。修改四路交叉路口的拐角半径。

1.  点击 **Corner Tool** 按钮 ![](https://ww2.mathworks.cn/help/roadrunner/ug/icon_tool_cornerce91ed4f4d96308ba6245d2db904f568.png)，然后点击以选择四路交叉路口。
    
    ![Intersection with four-way intersection selected](https://ww2.mathworks.cn/help/roadrunner/ug/gs_select_junction.png)
    
2.  默认情况下，连接点的角半径为 `5` 米。使用 **Attributes** 窗格增加此值。此窗格包含有关当前所选项目的信息和可编辑属性。在 **Corner Tool** 中，选择交叉口会选择交叉口的所有四个角，因此您可以同时修改所有四个角的属性。
    
    在 **Attributes** 窗格中，将所有四个角的 **Corner Radius** 属性设置为 `10`。
    
    ![Attributes pane of junction with Corner Radius set to 10](https://ww2.mathworks.cn/help/roadrunner/ug/gs_junction_attributes_pane.png)
    
    交叉口拐角在场景编辑画布中展开。
    
    ![Intersection with junction corners expanded](https://ww2.mathworks.cn/help/roadrunner/ug/gs_junction_corner_radius.png)
    

或者，您可以通过点击属性名称 ![Corner Radius attribute name selected](https://ww2.mathworks.cn/help/roadrunner/ug/gs_corner_radius_spin_box.png) 并向上或向下拖动来修改 **Corner Radius** 属性值。

### 添加人行横道

在交叉路口添加人行横道。

1.  旋转相机从上到下查看交叉路口。要将相机聚焦于选定的交叉路口，请按 **F** 键。
    
    ![Top-down view of intersection](https://ww2.mathworks.cn/help/roadrunner/ug/gs_pre_crosswalk.png)
    
2.  点击 **Crosswalk and Stop Line Tool** 按钮 ![](https://ww2.mathworks.cn/help/roadrunner/ug/icon_tool_crosswalk_and_stop_linecae12ba7b0814f777d62c08083e82568.png)。交叉路口显示蓝色 V 形，用于向交叉路口添加停止线。
    
    ![Intersection with blue chevrons that preview where stop lines are added](https://ww2.mathworks.cn/help/roadrunner/ug/gs_crosswalk_stoplines.png)
    
3.  从 **Library Browser** 中，选择一个人行横道添加到交叉路口。**Library Browser** 存储了可添加到场景的所有素材。素材包括三维对象、标记、纹理和材质。
    
    在 **Library Browser** 中，选择 `Markings` 文件夹，然后选择 `ContinentalCrosswalk` 素材。素材预览显示在素材查看器中。
    
    ![Library Browser with continental crosswalk asset selected](https://ww2.mathworks.cn/help/roadrunner/ug/gs_library_browser_crosswalk.png)
    
4.  在交叉路口内点击以清除蓝色 V 形。然后，右键点击交叉路口以将选定的人行横道素材应用到交叉路口。
    
    ![Intersection with crosswalk](https://ww2.mathworks.cn/help/roadrunner/ug/gs_crosswalk.png)
    

### 添加转弯车道

将交叉路口的其中一条道路转换为更复杂的高速公路，其中包括带箭头标记的转弯车道。

#### 改变道路风格

现有道路均采用默认道路样式，为简单的两车道分立式高速公路，设有人行道。更新交叉路口的其中一条道路以使用带有额外车道的道路样式。

1.  缩小并旋转相机，以类似于此处所示的角度查看场景。
    
    ![Scene viewed at an angle, with one of the intersecting roads facing the camera](https://ww2.mathworks.cn/help/roadrunner/ug/gs_road_style_original.png)
    
2.  在 **Library Browser** 中，打开 `RoadStyles` 文件夹，然后选择 `MainStreetCenterTurn` 素材。该道路样式素材包括路肩车道、每侧两条超车道和一条中间车道。（可选）在素材查看器中旋转和移动相机以检查道路样式。
    
    ![Library Browser with road style asset selected](https://ww2.mathworks.cn/help/roadrunner/ug/gs_road_style_asset.png)
    
3.  将选定的道路样式拖到最靠近相机的道路上，如下所示。道路更新为新样式并切换回道路规划工具。道路保持先前应用的拐角半径和人行横道样式。
    
    ![Road with new road style applied](https://ww2.mathworks.cn/help/roadrunner/ug/gs_road_style_montage.png)
    

#### 在交叉路口创建转弯车道

在交叉路口附近创建一条短的左转车道。

1.  旋转相机并放大具有新道路样式的道路一侧的人行横道附近。
    
    ![One side of intersection with the crosswalk at the top and the median lane at the center](https://ww2.mathworks.cn/help/roadrunner/ug/gs_marking_no_turning_lane.png)
    
2.  点击 **Lane Carve Tool** 按钮 ![](https://ww2.mathworks.cn/help/roadrunner/ug/icon_tool_lane_carve31d49515d2fa137e3797b4690a234d1f.png)。此工具使您能够在现有车道中创建锥形切口以形成转弯车道。
    
3.  点击以选择道路。然后，右键点击中间车道右侧要开始逐渐变细的位置。将蓝线对角拖动到中间车道的左侧，您希望在此结束锥形切口并开始转弯车道。
    
    ![Marking carve operation applied to median lane](https://ww2.mathworks.cn/help/roadrunner/ug/gs_marking_carve_montage.png)
    
4.  新形成的转弯车道仍保留中间车道的风格。更新车道标记以匹配标准转弯车道的样式。
    
    1.  在 **Library Browser** 中，选择 `SolidSingleWhite` 素材并将其拖到转弯车道的右侧。车道标记变为单白实线。
        
        ![Asset dragged onto right side of turning lane to change it into a solid single white line](https://ww2.mathworks.cn/help/roadrunner/ug/gs_marking_fix_markings1.png)
        
    2.  选择 `SolidDoubleYellow` 素材并将其拖动到形成转弯车道左侧的两个标记段上。车道标记线段变为双黄实线。
        
        ![Assets dragged onto left side of turning lane to change them into solid double yellow lines](https://ww2.mathworks.cn/help/roadrunner/ug/gs_marking_fix_markings_montage.png)
        
    
5.  在车道上添加一个转向箭头。在 **Library Browser** 的 `Stencils` 文件夹中，选择 `Stencil_ArrowType4L` 素材。将此素材拖动到转弯车道中要添加箭头模具的位置。
    
    ![Left arrow stencil dragged to bottom of turning lane](https://ww2.mathworks.cn/help/roadrunner/ug/gs_stencil1.png)
    
6.  通过添加箭头模板，RoadRunner 选择点标记工具 ![](https://ww2.mathworks.cn/help/roadrunner/ug/icon_tool_marking_pointddaf106f6dc9a3147e9b2fb02b0ce070.png) 使其成为活动工具。现在，您可以通过右键点击要添加第二个箭头的位置来添加它。
    
    ![Left arrow stencil copied above the previous stencil](https://ww2.mathworks.cn/help/roadrunner/ug/gs_stencil2.png)
    
7.  修改箭头的标记材质，使它们看起来更磨损。首先，选择两个箭头。在 **Library Browser** 的 `Markings` 文件夹中，选择 `LaneMarking2` 材质素材。然后，将该素材拖到所选箭头的 **Attributes** 窗格中，并覆盖现有的 `LaneMarking1` 材质素材。
    
    ![Lane marking texture dragged from Library Browser to the Attributes pane for the selected arrows](https://ww2.mathworks.cn/help/roadrunner/ug/gs_marking_material_montage.png)
    
    箭头更新为使用看起来更磨损的新材质。
    
    ![Turning arrows with new material applied](https://ww2.mathworks.cn/help/roadrunner/ug/gs_stencil3.png)
    

重复这些步骤以在交叉路口的另一侧创建转弯车道。

![Intersection with turning lanes on both sides](https://ww2.mathworks.cn/help/roadrunner/ug/gs_marking_complete.png)

### 添加道具

为了增强场景的细节，请添加道具。_道具_是可放置在道路上和周围的三维物体，例如支柱、电线杆和标志。使用多种技术在道路周围添加树木道具。

#### 添加单独的道具

将灌木丛添加到地形的一部分。

1.  缩小并旋转相机以适应整个道路网络和周围地形的视野。
    
    ![Scene with full road network and surrounding terrain in view](https://ww2.mathworks.cn/help/roadrunner/ug/gs_prop_original.png)
    
2.  在 **Library Browser** 中，打开 `Props` 文件夹并选择 `Trees` 子文件夹。
    
3.  选择灌木丛道具（以 `Bush_` 开头的素材文件之一）。将灌木丛拖到场景的一部分上。RoadRunner 切换到点道具工具 ![](https://ww2.mathworks.cn/help/roadrunner/ug/icon_tool_prop_point5a6b28965817f8314dd0296395b01024.png)。将其他灌木拖到场景中或右键点击以添加更多灌木。所有灌木丛均与地表地形对齐。
    
    ![Three bushes added to scene](https://ww2.mathworks.cn/help/roadrunner/ug/gs_prop_bushes.png)
    

#### 沿曲线添加道具

沿着曲线添加道具以遵循道路边缘。

1.  点击 **Prop Curve Tool** 按钮 ![](https://ww2.mathworks.cn/help/roadrunner/ug/icon_tool_prop_curve5e4066914262c663844da68d0f7b2ecd.png)。
    
2.  在 **Library Browser** 的 `Trees` 文件夹中，选择加州棕榈树道具（以 `CalPalm_` 开头的素材文件之一）。
    
3.  沿着交叉路口一侧的道路边缘右键点击，为其添加一行棕榈树。在远离曲线道具的地方点击以完成线条。
    
    ![A line of palm trees along one edge of the road](https://ww2.mathworks.cn/help/roadrunner/ug/gs_prop_span1.png)
    
4.  为了使跨度中的每棵树都可以移动和选择，您可以将曲线转换为单独的道具。选择曲线道具，然后在 **Attributes** 窗格中点击 **Bake**。棕榈树成为单独的道具，并且 RoadRunner 切换到点道具工具。将一些棕榈树移到交叉路口的另一侧。
    
    ![Palm trees converted to individual props and distributed along both sides of the intersection](https://ww2.mathworks.cn/help/roadrunner/ug/gs_prop_span2.png)
    

或者，要沿着道路跨度添加道具，您可以点击 **Prop Span Tool** 按钮 ![](https://ww2.mathworks.cn/help/roadrunner/ug/icon_tool_prop_span8e074421aaf180ebbe2c104859e7faa9.png)，选择一条道路，然后将道具拖到道路边缘上。

#### 在指定区域添加道具

在地面的指定区域添加道具。

1.  点击 **Prop Polygon Tool** 按钮 ![](https://ww2.mathworks.cn/help/roadrunner/ug/icon_tool_prop_polygond35cb3dc424ce356b920cb7af08121f0.png)。
    
2.  在 **Library Browser** 的 `Trees` 文件夹中，选择一个柏树道具（以 `Cypress_` 开头的素材文件之一）。
    
3.  右键点击地表地形的空白区域以绘制包含所选道具的多边形。点击远离多边形的位置以完成绘制。然后移动点或切线来改变多边形的形状。
    
    ![Cypress tree props added to a polygon. A tangent to the polygon is selected, which modifies the shape of the polygon.](https://ww2.mathworks.cn/help/roadrunner/ug/gs_prop_polygon.png)
    
4.  或者，使用 **Attributes** 窗格中的属性修改多边形道具。例如，要增加或减少多边形中的道具数量，请使用 **Density** 属性。要随机化多边形中的素材分布，请点击 **Randomize**。
    

#### 添加不同类型的道具

到目前为止，您已经向场景添加了一种类型的道具。要同时向场景添加多种道具，您可以创建道具集。

1.  在 **Library Browser** 的 `Trees` 文件夹中，按住 **Ctrl** 并选择您在前面部分添加到场景中的三个道具。
    
2.  选择 **New**，然后选择 **Prop Set**，并为道具集命名。新的道具组存储在 `Trees` 文件夹中。**Attributes** 窗格显示该套装中的三个道具和该道具集的预览。
    
    ![Attributes pane displaying a prop set containing a bush, palm tree, and cypress tree](https://ww2.mathworks.cn/help/roadrunner/ug/gs_prop_set1.png)
    
3.  点击 **Prop Polygon Tool** 按钮。在包含新道具集的地形空白部分创建多边形道具。
    
    ![Prop set added to terrain](https://ww2.mathworks.cn/help/roadrunner/ug/gs_prop_set2.png)
    
    或者，您还可以通过将道具集拖动到柏树的多边形上，将现有的柏树道具替换为新的道具集。
    

### 其他值得尝试的事情

您现在已经创建了一个简单的道路网络，其中包含真实的转弯车道、多个立交桥和不同类型的树木。

![Final RoadRunner scene](https://ww2.mathworks.cn/help/roadrunner/ug/gs_final_scene.png)

您现在可以使用其他工具增强场景。例如，尝试以下操作：

-   添加更多道路或连接场景中的现有道路。为了使车道数不同的道路之间的过渡更加平滑，请使用车道工具，例如车道工具、车道宽度工具、车道添加工具或车道形状工具。
    
-   使用信号工具在交叉路口添加交通信号灯。要修改每个转向信号处的车道路径，请使用操纵工具。有关示例，请参阅。
    
-   向场景添加额外的道具，例如桶、建筑物和交通标志。要修改标志的文字，请使用标志工具。
    
=======
>>>>>>> 8e86e410f27590e270dd9e096b10624f76bb261c

通过网盘分享的文件：1.zip
链接: https://pan.baidu.com/s/1JmsUv6jfPXRF-SCJmUjzkg?pwd=1aaa 提取码: 1aaa

## 使用方法
- **将文件解压缩到本地**：确保完整解压
- **找到文件地址**："1/Scenes/1.rrscene"，右键选择打开方式为roadrunner