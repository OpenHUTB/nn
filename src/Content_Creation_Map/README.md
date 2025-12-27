
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


通过网盘分享的文件：1.zip
链接: https://pan.baidu.com/s/1JmsUv6jfPXRF-SCJmUjzkg?pwd=1aaa 提取码: 1aaa

## 使用方法
- **将文件解压缩到本地**：确保完整解压
- **找到文件地址**："1/Scenes/1.rrscene"，右键选择打开方式为roadrunner

# 新建一个简单的roadrunner场景流程


![A simple RoadRunner scene](https://ww2.mathworks.cn/help/roadrunner/ug/gs_final_scene.png)


### 前提条件

在开始此示例之前，请确保您的系统满足以下前提条件：

-   您已安装并激活 RoadRunner。
    
-   您拥有 RoadRunner Asset Library 附加组件的许可证。此示例使用仅在该库中可用的素材。
    

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
好的，这是从 OpenStreetMap 网站获取 `.osm` 文件的完整、详细步骤的 Markdown 格式文档。

# 在RoadRunner中复刻现实道路的方法

## OpenStreetMap的网站

OpenStreetMap（简称OSM）是一个免费且开源的世界地图数据库，它允许用户查看、编辑和使用地理数据。通过OSM获取的数据可以导入到RoadRunner中，作为创建真实道路网络的基础。

**核心网站资源：**
- **主网站**：https://www.openstreetmap.org
  - 直接查看和浏览地图数据
  - 提供基础的导出功能（适合小区域）
- **数据提取平台**：https://overpass-turbo.eu
  - 专门用于查询和提取OSM数据
  - 支持更复杂的查询和更大的区域
  - 提供高级筛选功能

**数据特点：**
- 包含道路、建筑、自然要素等丰富信息
- 道路数据包含车道数、道路类型、限速等属性
- 完全免费且开放，适合商业和非商业用途

## 从OpenStreetMap获取OSM文件的步骤

### 方法一：通过OSM主网站直接导出

**适用场景：** 小范围区域（如单个路口、小型社区）

**详细步骤：**

1. **访问并定位**
   - 打开 https://www.openstreetmap.org
   - 使用鼠标滚轮缩放，左键拖拽，导航到目标区域

2. **调整显示范围**
   - 确保地图视图完全包含所需区域
   - 建议区域不超过2km×2km，避免超出导出限制

3. **执行导出操作**
   - 点击左侧工具栏的"导出"按钮
   - 地图上方会出现导出控制面板
   - 系统自动选择当前地图显示范围

4. **精细调整区域**（可选）
   - 点击"手动选择不同的区域"
   - 拖动矩形框的边角点调整选择范围
   - 确保只选择必要的道路区域

5. **下载文件**
   - 点击蓝色的"导出"按钮
   - 浏览器自动下载名为"map.osm"的文件
   - 保存到本地指定文件夹

**注意事项：**
- 如遇"请求的节点数过多"错误，需缩小选择范围
- 导出过程可能需要几秒到几分钟，取决于区域大小
- 确保网络连接稳定

### 方法二：通过Overpass Turbo获取（推荐）

**适用场景：** 大中型区域或需要筛选特定要素

**详细步骤：**

1. **打开Overpass Turbo**
   - 访问 https://overpass-turbo.eu
   - 等待地图和界面完全加载

2. **选择目标区域**
   - 使用地图缩放和平移找到目标区域
   - 点击工具栏的"瓦片选择"工具（方形带箭头图标）
   - 在地图上拖动鼠标框选精确区域

3. **构建查询语句**
   - 在左侧查询窗口中，使用以下基础查询：
   ```sql
   [out:xml][timeout:25];
   (
     node({{bbox}});
     way({{bbox}});
     relation({{bbox}});
   );
   out body;
   >;
   out skel qt;
   ```
   - 如需只获取道路数据，可使用：
   ```sql
   [out:xml][timeout:25];
   (
     way["highway"]({{bbox}});
   );
   out body;
   >;
   out skel qt;
   ```

4. **执行查询**
   - 点击"运行"按钮（三角形播放图标）
   - 等待查询完成，地图会高亮显示获取的数据
   - 检查数据完整性

5. **导出OSM文件**
   - 点击"导出"按钮（下载箭头图标）
   - 选择"原始OpenStreetMap数据"
   - 格式选择".osm"
   - 下载文件到本地


### 文件处理建议

1. **文件格式选择**
   - `.osm`：标准XML格式，兼容性最好
   - `.osm.pbf`：二进制格式，文件更小，适合大区域

2. **数据预处理**
   - 检查文件大小，过大的文件可能需要分割
   - 验证数据完整性
   - 备份原始文件

通过以上步骤获取的OSM文件，可以用于后续在RoadRunner中创建真实道路网络。建议先从小的测试区域开始，熟悉整个流程后再处理大型项目。


## 向roadrunner中导入osm文件的步骤
1. **打开roadrunner**
2. **打开或新建项目并选择目标场景（后缀为.rrscene的文件）**
  ![](https://github.com/2382613701lzy-netizen/image/blob/main/20251201112045_67_69.jpg)
3. **在上方工具栏找到SD Map Viewer Tool工具并选择（如下图圈注）**
  ![](https://github.com/2382613701lzy-netizen/image/blob/main/20251201112121_68_69.jpg)
4. **在左侧的工具栏找到Open Street Map File并打开**
  ![](https://github.com/2382613701lzy-netizen/image/blob/main/20251201112213_69_69.jpg)
5. **在弹出的窗口中找到从OpenStreetMap官网导出的后缀为.osm的文件打开**
  ![](https://github.com/2382613701lzy-netizen/image/blob/main/20251201112324_70_69.jpg)
6. **导入结果如下图**
  ![](https://github.com/2382613701lzy-netizen/image/blob/main/20251201112350_71_69.png)

# QGIS 下载、安装与中文界面设置完整指南
以下是QGIS软件从下载、安装到完成中文界面设置的完整步骤。整个过程主要分为下载、安装、设置语言以及获取学习数据四个部分。

### 第一步：下载QGIS
前往QGIS官方网站的下载页面 https://qgis.org/download/ 。页面会根据你的操作系统推荐合适的版本。
![](https://github.com/2382613701lzy-netizen/image/blob/main/20251216121655_88_69.png)
操作系统：软件支持 Windows、macOS、Linux、Android 和 iOS。

版本选择：

最新版：页面会提供最新的稳定版本（例如 QGIS 3.36.x），包含最新功能。

长期支持版 (LTR)：对于追求稳定性的用户或组织部署，推荐选择 QGIS 3.34 LTR，它经过了更长时间的测试。

Windows用户注意：官方自QGIS 3.20起，仅提供64位安装包。

### 第二步：安装QGIS
根据你的操作系统，参考以下步骤完成安装。

#### *Windows*	

安装步骤
1. 运行下载的 .exe 安装程序。
2. 在安装类型界面，普通用户建议选择“标准安装(Standard Install)”。
3. 按照安装向导提示完成后续步骤。
#### *macOS*	
1. 双击下载的 .dmg 磁盘映像文件。
2. 将QGIS图标拖拽到“应用程序(Applications)”文件夹中。
3. 首次启动需右键点击应用并选择“打开”，以通过系统安全验证。
#### *Linux*	
根据你的发行版，使用对应的软件包管理器（如apt、yum）或从官方仓库安装。
### 第三步：设置为中文界面
QGIS内置了多语言支持，切换为中文无需安装额外语言包。

打开QGIS，在顶部菜单栏点击 Settings (设置) → Options (选项)。

在弹出的选项窗口中，在左侧选择 General (常规) 选项卡。

在右侧找到 Locale (区域设置) 板块，你会看到 Override system locale (覆盖系统区域设置) 的选项。

勾选此选项，然后在下方 Language (语言) 的下拉菜单中，选择 中文(简体) 或 中文(繁体)。

点击右下角的 OK (确定) 保存设置。

完全关闭并重新启动QGIS，界面就会切换为中文了。

注：如果重启后界面未改变，请检查是否成功保存了设置，并确认在Language下拉菜单中正确选择了“中文(中国)”。

最后的应用界面如下图说明你的QGIS已经成功配置可以使用了
![](https://github.com/2382613701lzy-netizen/image/blob/main/20251216121909_89_69.png)


# 天地图数据获取与RoadRunner场景构建全流程教程

本教程旨在指导你如何从天地图获取卫星影像数据，并将其作为精确参考底图导入专业道路场景建模软件RoadRunner，用于创建高真实度的仿真场景。

## 第一步：前置准备 —— 获取天地图访问密钥

在开始任何操作前，你必须拥有一个有效的天地图服务密钥。

1.  **访问官网**：前往 [天地图官网](http://lbs.tianditu.gov.cn/home.html) 注册并登录。
2.  **创建应用**：进入控制台的 **“应用管理”**，点击 **“创建新应用”**。
3.  **关键设置**：
    *   **应用类型**：**务必选择 “浏览器端”**。
    *   **IP地址**：为保持使用灵活性，建议**留空不填**。若需填写，格式必须为 `http://你的IP或域名`。
4.  **获取密钥**：应用创建成功后，系统会生成一个唯一的密钥（一串字符串）。请妥善保存，后续所有步骤都将用到它。


![](https://github.com/2382613701lzy-netizen/image/blob/main/20251216103950_83_69.png)
*注意：这是我自己的密钥，请使用你自己自己创建应用成功后的密钥并且妥善保管*

## 第二步：核心操作 —— 在QGIS中获取并导出底图

本部分使用QGIS作为桥梁，将在线天地图服务“固化”为本地地理图像文件。

### 2.1 在QGIS中添加天地图服务
1.  **安装并打开QGIS**。
2.  **新建连接**：在左侧“浏览器”面板，右键点击 `XYZ Tiles`，选择 **新建连接**。
3.  **填写连接信息**：
    *   **名称**：可自定义，如 `天地图影像`。
    *   **URL**：粘贴以下地址，并务必将 `你的密钥` 替换为你在第一章获取的真实密钥。
    ```
    https://t0.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=你的密钥
    ```

    ![](https://github.com/2382613701lzy-netizen/image/blob/main/20251216110003_84_69.png)
5.  **加载图层**：点击“确定”后，双击此新建连接，天地图卫星影像将加载到主地图窗口。
    ![](https://github.com/2382613701lzy-netizen/image/blob/main/20251216110316_85_69.png)
> **可选：添加注记图层**
> 如需显示道路名、地名等文字信息，请按上述步骤再添加一个连接，URL如下：
> ```
> https://t0.tianditu.gov.cn/cia_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=cia&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=你的密钥
> ```

### 2.2 定位并设置清晰度
1.  **导航到目标区域**：使用鼠标滚轮缩放和平移，找到你要创建场景的具体位置。
2.  **选择合适缩放级别**：**缩放级别直接决定最终图像的清晰度**。状态栏会显示当前级别，建议参考：
    *   **城市/区域范围**：13-15级
    *   **街区/园区范围**：16-17级
    *   **路口/建筑细节**：18-19级

 ![](https://github.com/2382613701lzy-netizen/image/blob/main/20251216111009_86_69.png)

### 2.3 导出为地理配准图像（关键步骤）
此步骤将在线地图转换为RoadRunner可读的本地文件。

1.  点击菜单栏：**项目 → 导入/导出 → 导出地图为图片...**。
2.  在保存对话框中：
    *   **选择保存位置和文件名**。
    *   **格式选择**：在下拉菜单中选择 **`TIFF`**。该格式即是带地理信息的 **GeoTIFF**。
3.  **设置参数（可选但重要）**：
    *   **分辨率 (DPI)**：建议设置为 **300** 或更高，以获得清晰图像。
    *   **范围**：选择“使用地图画布范围”以导出当前视图，或点击“选择在地图上”手动框选。
4.  点击 **保存**。QGIS将开始渲染并生成最终的 `.tiff` 文件。

## 第三步：场景构建 —— 在RoadRunner中导入与配准

### 3.1 导入底图
1.  启动RoadRunner并打开你的项目。
2.  将上一步导出的 `.tiff` 文件直接从文件资源管理器**拖拽**到RoadRunner的 **Library Browser** 或主视图窗口。
 ![](https://github.com/2382613701lzy-netizen/image/blob/main/20251217124002_92_69.jpg)
 ![](https://github.com/2382613701lzy-netizen/image/blob/main/20251217124003_93_69.jpg)

### 3.3 开始场景创作
配准完成后，天地图底图将作为**静态参考背景**。后续所有三维创作均需基于此背景**手动完成**：

*   **绘制路网**：使用道路工具，**严格沿底图上的道路影像边缘描绘**，生成车道、路口。
*   **摆放资产**：参照底图，将建筑、树木、交通标志等三维模型放置在正确位置。
*   **地形塑造**：根据影像中的地貌起伏，调整地形高度。

## 第四章：进阶技巧

### 4.1 提高效率：使用矢量数据辅助路网创建
完全手动描路效率较低。可以寻找道路矢量数据导入RoadRunner作为辅助参考线。
*   **推荐数据源**：**开放街道地图 (OpenStreetMap, OSM)** 提供免费、全球覆盖的道路矢量数据，可通过 [OSM官网](https://www.openstreetmap.org) 或QGIS的 **QuickOSM** 插件方便地导出所需区域数据。

### 4.2 常见问题排查
*   **QGIS连接显示“Forbidden”**
    1.  检查密钥是否过期，确保天地图控制台内应用状态为“正常”。
    2.  **确认应用类型为“浏览器端”**。
    3.  核对URL，确保 `tk=` 参数后的密钥填写无误。
    4.  尝试将URL中的 `t0.tianditu.gov.cn` 替换为 `t1` 到 `t6` 中的其他域名。
*   **导出的图像模糊**
    1.  在QGIS导出前，请确保已缩放至**足够高的级别**（如17级以上）。
    2.  导出时请将分辨率（DPI）设置为300或更高。

### 4.3 理解数据用途
请务必了解：本教程方法获取的天地图数据，在RoadRunner中**仅作为视觉参考底图**，**不会自动生成任何三维模型**。所有道路曲面、车道线、交通设施、建筑模型等，都需要你利用RoadRunner的强大工具手动创建和摆放。其核心价值在于提供了 **“在真实坐标框架下进行精确设计”** 的能力。

# 进阶：考虑使用点云来辅助创建地图
在 RoadRunner 中集成点云数据以辅助高精度三维场景构建，已被证明是一条兼具工程便捷性与技术可行性的高效路径。该方法的核心价值在于将离散的真实世界测量数据，转化为结构化虚拟环境的几何与语义基础，从而显著提升场景建模的精度、效率及下游应用价值。

### 核心优势：精度与效率的闭环

在工程实践中，此工作流的核心优势体现在两个层面：
1.  **数据驱动的精度保障**：激光雷达点云提供了厘米级乃至毫米级的三维空间采样，为道路曲率、坡度、路缘石高度、建筑立面等关键地理要素提供了不可替代的几何真值参考。在 RoadRunner 中，建模操作可直接以点云为空间基准进行对齐与拟合，从根本上确保了生成场景的度量精度，满足了自动驾驶仿真等应用对空间保真度的苛刻要求。
2.  **流程化的效率提升**：点云提供了直观的三维空间上下文，将建模人员从依赖二维影像进行空间推理的抽象工作中解放出来。道路标定、地形生成、建筑物轮廓提取等任务，转变为在可视化三维参考下的交互式操作，大幅减少了反复测量与修正的迭代周期，降低了复杂场景（如大型立交枢纽、不规则地形）的构建门槛。

### 技术可行性：多源集成与标准输出

该方法的可行性，由 RoadRunner 平台强大的数据兼容性与标准化输出能力所支撑：
*   **多源数据融合**：平台支持点云与航空正射影像、数字高程模型、矢量路网等多源地理信息数据的同步加载与联合显示。这允许建模人员在统一的时空参考系中，综合利用不同数据源的优势——例如，以影像辅助纹理与语义判断，以点云约束几何形态，以矢量数据定义拓扑关系——进行交叉验证与协同建模，构建出信息完备的场景。
*   **下游生态兼容性**：基于此流程创建的场景模型，并非孤立的静态资产。其内嵌的车道模型、交通标志、信号灯等均已包含符合行业标准的语义属性。通过直接导出为 **OpenDRIVE**（描述道路逻辑）、**FBX** 或 **glTF**（描述三维模型与外观）等标准格式，生成的环境可直接接入 **CARLA**、**VTD**、**百度 Apollo** 等主流自动驾驶仿真与测试框架，实现了从“数据到模型，再到仿真应用”的端到端贯通，验证了其工业级应用的成熟度。

### 实施考量与最佳实践

为充分发挥该工作流的潜力，在实施中需关注以下技术要点：
*   **数据预处理是基础**：成功集成的首要前提是确保点云数据具备精确且一致的地理坐标系与投影信息。在导入前，通常需对原始点云进行必要的预处理，如坐标转换、无效点过滤、以及针对大规模数据的适度分块，以确保其在 RoadRunner 场景中能够正确空间配准。
*   **性能与精度平衡**：面对超大范围的高密度点云，需在数据精度与软件交互流畅性之间取得平衡。实践中，可根据当前建模的细节等级需求，选择使用点云的全分辨率版本，或通过抽稀处理后的轻量化版本作为参考背景，以实现更高效的操作体验。

在 RoadRunner 中引入点云数据，实质上是构建了一条 **“物理感知-数字建模-虚拟仿真”** 的数字化管道。它不仅是提升单点建模效率的工具，更是确保虚拟环境在几何、语义和行为层面与真实世界保持一致的关键技术环节，为高可靠性的自动驾驶系统测试与验证提供了至关重要的环境基础。

# 从何处获取点云数据

Sydney Urban Objects Dataset

链接: [Sydney Urban Objects Dataset](https://www.acfr.usyd.edu.au/papers/data.shtml)

介绍: 由悉尼大学提供的城市道路物体点云数据集，使用Velodyne HDL-64E LIDAR在悉尼中央商务区收集。包含车辆、行人、广告牌等分类信息，适用于自动驾驶技术的研究。

Large-Scale Point Cloud Classification Benchmark (Semantic3D)

链接: [Semantic3D](http://www.semantic3d.net/)

介绍: 包含了大量标记的城市环境3D点云数据集，适合大规模点云分类研究。涵盖了从教堂到铁路轨道等多种场景，数据量大且标注详细。

KITTI Vision Benchmark Suite

链接: [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)

介绍: 主要面向自动驾驶领域，提供了丰富的传感器数据，包括点云、图像、GPS和IMU数据。数据来源于真实世界的驾驶场景，是评估自动驾驶系统性能的重要资源。

OpenTopography

链接: [OpenTopography](https://opentopography.org/)

<img width="1920" height="1150" alt="364f1c173de0e829083dc59dbcce0418" src="https://github.com/user-attachments/assets/20e523a0-64b5-4d6c-a15a-dcec014be906" />

介绍: 提供全球范围内的高分辨率地形数据，包括激光雷达（LiDAR）点云数据。适合地理信息系统（GIS）、地貌学和其他地球科学领域的研究人员使用。

USGS National Map 3DEP Program

链接: [](https://apps.nationalmap.gov/downloader/)

<img width="1920" height="1149" alt="image" src="https://github.com/user-attachments/assets/6db55a2f-d0e5-4d9c-8abc-2d72d78bdfcd" />


介绍: 美国地质调查局提供的全国3D高程计划（3DEP），包含了美国本土的LiDAR点云数据，可用于洪水建模、水资源管理等多个领域。

# 以美国国家地质调查局（USGS）为例

## 需要说明的是，USGS显然不能提供中国地区的点云数据下载，事实上，USGS只能提供美国的大部分地区的雷达扫描记录的点云数据下载。当然，作为roadrunner练手的素材获取渠道，USGS是完全够用的。

**1.打开[](https://apps.nationalmap.gov/downloader/)**

<img width="1910" height="1094" alt="image" src="https://github.com/user-attachments/assets/c9395f59-a521-4f34-ae83-4057c636dce6" />

**2.通过鼠标的拖拽和缩放将你需要获取数据的区域大致移到你的屏幕中央**

**3.在左侧的“地图数据”勾选高程源数据选项**

<img width="1910" height="1094" alt="image" src="https://github.com/user-attachments/assets/a580ee02-7a98-48e9-b17e-d53e98888936" />

**4.滚动滚轮，在下方勾选“Lidar Point Cloud (LPC)”和“ LAS,LAZ”项**

<img width="755" height="550" alt="image" src="https://github.com/user-attachments/assets/84d371c4-62fb-4553-8308-dff4b0d26a8b" />

**4.在左上方勾选你用来标定具体区域的方法，此处我以“范围”为例**

<img width="1910" height="1094" alt="image" src="https://github.com/user-attachments/assets/7c67a50a-a4c4-41eb-b5ea-8265b6b1929d" />

**5.搜索产品**

<img width="1910" height="1094" alt="image" src="https://github.com/user-attachments/assets/51bffa50-40ec-4dde-b0fd-e7023d24f74c" />

**6.下载数据**

<img width="580" height="941" alt="image" src="https://github.com/user-attachments/assets/0e52a073-0c79-44c0-b95a-8045f2599fe5" />

在左侧选中需要的数据后点击Download Link (LAZ)进行下载

**7.注意此时你需要想办法获取这块区域的EPSG编码（后面可能会用上）**

你需要找到如图所示的“Info/Metadata”项
<img width="790" height="866" alt="image" src="https://github.com/user-attachments/assets/b3967b65-3583-453c-9ae5-1f893f4c7fdd" />
跳转到
<img width="1910" height="1094" alt="image" src="https://github.com/user-attachments/assets/f5226e1a-8fb2-4eeb-a2c9-5185326db45b" />
往下翻找到
<img width="1910" height="1094" alt="image" src="https://github.com/user-attachments/assets/c59896c3-40d9-440d-ab2b-ab69e5c5b810" />
点击后往下翻重点记录下来这部分
<img width="1409" height="1090" alt="image" src="https://github.com/user-attachments/assets/77e2e8c2-1a0e-45d1-b794-028c315570ee" />
然后可以去查对应的EPSG编码


# 将点云文件导入roadrunner的方法

## 第一步：将.laz文件解压为.las文件

### 方法一：使用laszip进行解压
将.laz文件解压为.las文件需要用到laszip，你可以在GitHub找到LAStools，LAStools 是一组高效、多核批量、可脚本化的工具，用于处理 LAS、压缩 LAZ、Terrasolid BIN、ESRI Shapefile（SHP）、ASCII 等。仓库链接为：[](https://github.com/LAStools/LAStools)
<img width="1910" height="1094" alt="image" src="https://github.com/user-attachments/assets/13d1c472-8da7-44b5-8d28-300f72dc4a7a" />
1.**在你的本地创建一个文件夹作为本地仓库并初始化**
进入文件夹右键空白处，“在终端打开”→“输入git init”→“完成初始化”
2.**克隆LAStools到本地仓库**
在打开的终端输入
git clone https://github.com/LAStools/LAStools.git
等待克隆完成
<img width="1700" height="990" alt="image" src="https://github.com/user-attachments/assets/06699df4-bd84-4eb1-b1b9-0afa57f100fe" />
3.**找到laszip并运行**
文件路径："你的仓库\LAStools\bin\laszip.exe"
程序UI：<img width="1275" height="991" alt="image" src="https://github.com/user-attachments/assets/ef2d27ef-48a4-48b1-ae60-1ab206987755" />
4.**使用laszip将.laz文件解压为.las文件**
首先你需要把.laz文件移到仓库所在的文件夹下
然后回到laszip
<img width="1275" height="989" alt="image" src="https://github.com/user-attachments/assets/cec19d46-7b1c-404a-9a11-304de883b208" />
<img width="1280" height="959" alt="image" src="https://github.com/user-attachments/assets/ef5edfa2-0d0c-4a65-9b0f-c2b73b6a6382" />
<img width="1274" height="984" alt="image" src="https://github.com/user-attachments/assets/6101384e-0963-498c-adce-863b24392f38" />
回到.laz文件所在的那一级文件夹，可以看到生成了一个同名的.las文件，此时你已经成功的将.laz文件转为.las文件

### 方法二：使用QGIS内置的工具进行格式转换

1.**打开QGIS**

菜单栏->数据处理->工具箱
<img width="1920" height="1198" alt="image" src="https://github.com/user-attachments/assets/fccc9043-c816-49a8-bbb7-10c7488fe8c9" />
<img width="1920" height="1198" alt="image" src="https://github.com/user-attachments/assets/7dd0759c-905d-4b6c-a221-413dd76e69d5" />
2.**选择格式转换**

点云转换->格式转换，选中格式转换（双击）
<img width="1920" height="1169" alt="image" src="https://github.com/user-attachments/assets/fe4e07a0-a60f-4e45-9e8f-41a464ffd3f4" />
3.**进行格式转换**

<img width="860" height="681" alt="image" src="https://github.com/user-attachments/assets/eedfddae-05ee-4833-89af-7b2438e4c347" />

4.**转换完成**

<img width="860" height="654" alt="image" src="https://github.com/user-attachments/assets/5cc6fb83-5bfd-4799-9725-2ef7312b24c0" />


# 将.las/.laz文件导入roadrunner

**1.打开roadrunner新建一个场景**

## *注意：新建场景的路径不能有中文字符存在，否则会报错*

<img width="615" height="314" alt="image" src="https://github.com/user-attachments/assets/b4b7d71c-0df9-43b2-a22c-6729fadc08fa" />

**2.将.las/.laz文件直接拖入roadrunner的Library Browser**

**3.将.las/.laz文件从Library Browser拖入场景**

**成功**

<img width="1920" height="1169" alt="image" src="https://github.com/user-attachments/assets/afecb182-3bba-4884-9b58-e47abd118553" />

## 常见的错误的解决方案

如果报错为

<img width="615" height="314" alt="image" src="https://github.com/user-attachments/assets/b4b7d71c-0df9-43b2-a22c-6729fadc08fa" />

有时不是因为路径中出现了中文字符，而是你的.las/.laz文件缺少正确的坐标系信息。

### 解决方案：


1.**找到上文提到的对应区域的EPSG编码**

确保你的点云文件已经存在roadrunner的 Library Browser中

2**打开Attributes窗口**

![2c6b2cdf01ada25e05e353659654fe98](https://github.com/user-attachments/assets/5aef7363-91b5-4b90-ab76-712ed3f9471f)

2.**选中.las/.laz文件并查看Attributes**

在Attributes窗口中

![b24259663a09b40d94ef0e92af6126de](https://github.com/user-attachments/assets/f1536f8e-62a4-485c-94ad-825453dd6cd1)

3.**在QGIS中导入.las/.laz文件查看坐标信息**

记下坐标信息之后返回roadrunner，选中world settings tool
<img width="1920" height="1200" alt="_cgi-bin_mmwebwx-bin_webwxgetmsgimg__ MsgID=697152194148194368 skey=@crypt_38968688_e13c903e415bb13501591a3aeadbfff7 mmweb_appid=wx_webfilehelper" src="https://github.com/user-attachments/assets/4c38ed1f-a446-4a51-861a-bb608a924ce6" />


根据坐标信息进行世界原点的设置

# 导出至 CARLA

## CARLA 导出概览

RoadRunner 可以将场景导出到 CARLA 模拟器。有两个 CARLA 导出选项可供选择：

*   **CARLA**：导出 **Unreal® Datasmith (`.udatasmith`)** 文件和 **ASAM OpenDRIVE® (`.xodr`)** 文件。
*   **CARLA Filmbox**：导出一个 **Filmbox (`.fbx`)** 文件、一个包含部分元数据的 **XML** 和一个 **ASAM OpenDRIVE (`.xodr`)** 文件。XML 文件保存场景中材质的数据。

在 CARLA 或 Unreal 端，提供了一个插件来帮助导入从 RoadRunner 导出的场景。

**对于使用CARLA 插件导出的场景**，Unreal 端提供的插件处理以下内容：
*   **设置材质**
    *   材质数据从 Datasmith (`.udatasmith`) 文件中读取，并将数据映射到插件中包含的基础材质之一的新实例中。
    *   透明材质将根据漫反射颜色的透明度在半透明和遮罩混合模式之间进行选择。
*   **调整导入的静态网格物体中的碰撞器**
    *   在导入过程中，新创建的静态网格素材的 "Collision Complexity" 选项设置为 "Use Complex Collision As Simple"。
*   **设置交通信号视觉效果**
    *   交通信号逻辑连接到模拟器。
*   **软件要求**
    *   CARLA 0.9.13

**对于使用CARLA Filmbox 插件导出的场景**，Unreal 端提供的插件利用 XML 文件中存储的信息帮助导入 FBX® 文件。该插件处理以下内容：
*   **设置材质**
    *   从 XML 文件中读取材质数据，并将数据映射到插件中包含的基础材质之一的新实例中。
    *   某些材质将从 CARLA 材质之一中实例化。
    *   透明材质将根据漫反射颜色的透明度在半透明和遮罩混合模式之间进行选择。
*   **调整导入的静态网格物体中的碰撞器**
    *   在导入过程中，新创建的静态网格素材的 "Collision Complexity" 选项设置为 "Use Complex Collision As Simple"。
*   **设置交通信号视觉效果**
    *   交通信号逻辑未连接到模拟器。
*   **软件要求**
    *   CARLA 0.9.13

---

## 安装插件

按照本节中的说明安装 Unreal 插件：

1.  从源代码构建 CARLA。有关更多信息，请参阅构建 CARLA 说明的 Windows® 构建页面。
2.  有关下载最新版本插件的说明，请参阅 下载插件。
3.  解压 RoadRunner 插件 zip 文件并找到 `RoadRunnerImporter`、`RoadRunnerCarlaIntegration`、`RoadRunnerRuntime`、`RoadRunnerDatasmith`、`RoadRunnerCarlaDatasmith` 和 `RoadRunnerMaterials` 文件夹，它们位于 "Unreal/Plugins" 下。
4.  复制 `RoadRunnerImporter`、`RoadRunnerCarlaIntegration`、`RoadRunnerRuntime`、`RoadRunnerDatasmith`、`RoadRunnerCarlaDatasmith` 和 `RoadRunnerMaterials` 文件夹，放入工程目录 `Plugins` 文件夹下的 `CarlaUE4` 文件夹中，位于 `<carla>/Unreal/CarlaUE4/Plugins`（Carla 文件夹旁边）。

<img width="302" height="185" alt="download-new-plugins" src="https://github.com/user-attachments/assets/8aa371b6-062e-4be3-b6a6-99734277113b" />

5.  重建插件。首先，生成工程文件。
    *   如果您在 Windows 上，请右键点击 `.uproject` 文件并选择 **Generate Visual Studio project files**。
    *   如果您使用 Linux®，请在命令行运行以下代码：
        ```
        $UE4_ROOT/GenerateProjectFiles.sh -project="<CarlaFolderPath>/Unreal/CarlaUE4/CarlaUE4.uproject" -game -engine
        ```
        将 `UE4_ROOT` 设置为 Unreal Engine® 安装目录。

    然后，打开工程并构建插件。如果您使用的是 Windows，请在 VS 2019 的 x64 本机工具命令提示符中运行 "make launch" 来编译插件并启动编辑器。
6.  插件显示在 **Edit > Plugins** 下。如果它没有出现在该菜单中，请检查 **Enabled** 复选框是否选中。

<img width="571" height="577" alt="new-plugin-list" src="https://github.com/user-attachments/assets/febc75c8-6db5-4b71-867b-36c62947c4a9" />

### 插件内容

*   **RoadRunnerImporter 模块**：
    *   当元数据文件存在时，覆盖默认的 FBX 导入器
    *   使用元数据文件用新材质覆盖默认材质的选项
    *   导入信号数据和时序
*   **RoadRunner 运行时模块**：
    *   包含控制交通信号视觉效果的组件
*   **RoadRunnerCarlaIntegration 模块**：
    *   创建一个新地图并将 FBX 导入到关卡中
    *   根据分割类型移动静态网格物体素材
    *   创建由 CARLA 材质实例化的材质，用于天气效果
    *   从 ASAM OpenDRIVE 文件生成路线
*   **RoadRunnerMaterials 插件**：
    *   用于创建实例的基础材质
*   **RoadRunnerDatasmith 插件**：
    *   处理元数据后处理的 Dataprep 素材
    *   导入信号数据和时序
*   **RoadRunnerCARLADatasmith 插件**：
    *   将 RoadRunner 场景导入 CARLA，自动设置交通信号灯。

---

## 从 RoadRunner 导出至 CARLA

### 使用 CARLA 插件导出

**CARLA 插件（.udatasmith + .xodr）是导出到 CARLA 的推荐方法。** 使用 CARLA (`.udatasmith` + `.xodr`) 插件将 RoadRunner 场景导出到 CARLA，您可以将 ASAM OpenDRIVE 数据与 Datasmith 文件一起导出，并将元数据添加到 Datasmith 导出中。元数据存储信号和 ASAM OpenDRIVE ID。CARLA 导出选项减少了将大型场景导入 CARLA 的时间。

按照以下步骤使用 **CARLA（.udatasmith + .xodr）** 将场景从 RoadRunner 导出到 Unreal：

1.  在 RoadRunner 中打开您的场景。
2.  使用 **CARLA (`.udatasmith` + `.xodr`)** 选项导出场景。从菜单栏中选择 **File > Export > CARLA (.udatasmith, .xodr)**。
3.  在 "Export CARLA Road" 对话框中，设置网格合并和平铺选项，然后点击 **Export**。
4.  选择 **Browse** 打开文件对话框，设置导出文件的名称和路径。Datasmith 文件导出到指定文件夹。
    *   您可以按分割类型划分网格。网格体的名称后附加有 `<segmentation type>Node`。
    *   如果选择了 **Export To Tiles** 选项，网格将按图块进行分割。道具按它们所在的图块进行分组。
        *   默认情况下，仅导出一个文件。图块存储在单独的节点中。
        *   如果启用了 **Export Individual Tiles**，则每个图块都会存储在其自己的 Datasmith 文件中。

### 使用 CARLA Filmbox 导出

如果您想使用较旧的管道进行导出，请使用 **CARLA 插件（.fbx + .rrdata.xml + .xodr）** 选项。按照以下步骤使用 **CARLA Filmbox 插件 (.fbx + .rrdata.xml + .xodr)** 选项将场景从 RoadRunner 导出到 CARLA：

1.  在 RoadRunner 中打开您的场景。
2.  使用 **CARLA** 选项导出场景。从菜单栏中选择 **File > Export > CARLA Filmbox (.fbx, .xodr, .rrdata.xml)**。
3.  在导出 CARLA 对话框中，根据需要设置 **FBX** 选项卡上的网格平铺和 **OpenDRIVE** 选项卡上的 ASAM OpenDRIVE 选项。然后，点击 **Export**。
4.  浏览打开文件对话框以设置导出文件的名称和路径。FBX、纹理、XML 和 ASAM OpenDRIVE 文件导出到同一文件夹。
    *   网格可以按分割类型进行划分。网格有 `"<segmentation type>Node"` 附加到它们的名字后面。
    *   如果选择了 **Export To Tiles** 选项，网格会按图块分割，道具会按其所在的图块分组。
        *   默认情况下，仅导出一个文件。图块存储在单独的节点中。
        *   如果启用了 **Export Individual Tiles**，则每个图块都会存储在其自己的 FBX 文件中。

> **注意**
> 该插件不完全支持 **Export Individual Tiles** 选项。

---

## 导入 CARLA

### 导入到 CARLA

如果您使用 **CARLA (`.udatasmith` + `.xodr`)** 选项导出了 RoadRunner 场景，请按照以下步骤将场景导入 CARLA：

1.  复制 **BaseMap** 并将其保存到 `Maps` 文件夹。重命名新地图。

<img width="547" height="306" alt="basemap" src="https://github.com/user-attachments/assets/68a03ab6-43f3-407e-818c-f5fa7cdd2088" />

2.  打开您在上一步中创建的新地图。
3.  在 Unreal 中的 **Content Browser** 窗口中右键点击。在菜单中，选择 **Show Plugin Content**。

<img width="152" height="300" alt="content-browser" src="https://github.com/user-attachments/assets/0cc1dfb8-4fa6-418e-bc7b-637c1c488c66" />

4.  在 CARLA 中启用 **Datasmith FBX Importer** 和 **Datasmith Importer** 插件。
5.  点击 **RoadRunner CARLADatasmith Content**。双击 **RRCARLADataprep Asset**。这将打开一个新的编辑器，用于处理使用 Datasmith 的导入以及所有后期处理步骤。

<img width="780" height="227" alt="new-carla-plugin" src="https://github.com/user-attachments/assets/789359ae-6ae1-4c1c-870f-0451f630693f" />

6.  点击 **Import** 按钮将场景导入 Unreal。例如，此图像加载 `FourWaySignal` 场景文件，它是 RoadRunner 工程的 `Scenes` 文件夹中存在的场景之一。

<img width="1968" height="1187" alt="imported-scene-datasmith" src="https://github.com/user-attachments/assets/ec5b11fc-0ee7-40c9-8817-d365f7987fbc" />

7.  点击编辑器工具条中的 **Execute** 即可运行导入场景的后期处理步骤。
8.  点击编辑器工具条中的 **Commit** 将这些更改提交到场景。
9.  场景现已导入并准备进行模拟。

### 导入至 CARLA (Filmbox)

如果您在导出 RoadRunner 场景时使用了 **CARLA Filmbox** 插件，请按照以下步骤操作。

将文件拖到内容浏览器中。
    *   使用 "Import" 按钮并选择 FBX 文件。
    *   该插件检查是否存在与导入的文件相关联的 RoadRunner XML 文件，如果未找到相应的 XML 文件，则正常导入。
    *   选择 **File > Import Into Level** 不会使用导出的 RoadRunner XML，而是使用 Unreal 导入器。
**当RoadRunner 导入选项对话框打开时：**
<img width="400" height="246" alt="import_to_unreal_pop_up" src="https://github.com/user-attachments/assets/8e1000ce-dea0-4202-a9bd-6ca736d253e3" />

* 覆盖材质

    * 覆盖默认材质导入。来自 CARLA 材质的道路和树叶实例。

    * 如果您想在下一个对话框中将材质设置为 Use Existing，则需要取消选中。

* 导入信号视觉效果

    * 仅当在下一个对话框中选择 "Create one Blueprint asset" 选项时才起作用。
        > 导入信号视觉效果选项对交通模拟没有任何影响。
      > 
**当 FBX 场景导入选项对话框打开时：**

1.将 **Scene > Hierarchy Type** 设置为 **"Create One Blueprint Asset"**（默认选择）。
        
**注意**
        
> 只有 **"Create One Blueprint Asset"** 导入选项适用于材质、信号和透明度排序。**"Create one Actor with Components"** 和 **"Create Level Actors"** 选项仅导入材质。

2.如果需要，请选择 **Invert Normal Maps**。

<img width="861" height="436" alt="exporting_to_carla_03" src="https://github.com/user-attachments/assets/7831b3e0-c842-4bb7-a191-f6560c6bc900" />

3.将 **Static Meshes > Normal Import Method** 设置为 **Import Normals**。
<img width="829" height="436" alt="exporting_to_carla_04" src="https://github.com/user-attachments/assets/c1d67091-b824-4f3e-b684-899ca1d9d167" />

4.（可选）取消勾选 **Remove Degenerates**，这可以帮助制作一些较大规模的道具。

5.点击 **Import**。

**关于将交通信号导入 Unreal：** 如果在 RoadRunner 中设置了交通信号灯，则它们将作为 `RoadRunnerTrafficJunction` 组件导入到 Unreal 中。这些控制器在导入期间自动创建，并包含在创建的蓝图中。`RoadRunnerTrafficJunction` 组件处理信号状态之间切换的逻辑。UUID 用于匹配场景中的特定游戏对象。

**FBX 详细信息：** 由于以下原因，FBX 文件将通过分割和透明度排序层自动划分网格：
*   **分割**：CARLA 通过静态网格素材确定分割。
*   **透明度排序**：Unreal 将 "Translucency Sort Priority" 值存储在静态网格组件上。

### 测试地图

1.  在编辑器中点击 **Play**（第一次点击 Play 需要额外时间来构建地图）。
<img width="326" height="158" alt="exporting_to_carla_05" src="https://github.com/user-attachments/assets/b60b6a7c-69e9-4a30-9c68-6bc8906c16d5" />

2.  运行示例 Python® 脚本。
<img width="1149" height="592" alt="test-map" src="https://github.com/user-attachments/assets/4a891fa1-62d6-4421-970c-53a521b2e222" />

# 使用ros进行桥接（封装）
## 第一步：本指南完成ros封装操作的前置条件：
1.**Ubuntu版本为20.04**

2.**ros版本为ros1**

3.**Carla版本为源码编译的0.9.13**

  注意！预编译包安装的Carla通常完成不了完整导入整个地图（包括建筑和材质）到Carla中的功能，需要采用源码编译的方法为你的操作系统配置上Carla，用源码编译对于内存和磁盘空间的要求较高，通常需要内存大于或等于16G、空闲磁盘空间大约200G。
  
## 第二步：参考上面的“将roadrunner生成的的地图文件导入Carla”教程成功在Carla中完美再现你所建造的地图 ##

## 第三步：编译 ROS Bridge ##

首先，请按官方指南从源码编译 carla-ros-bridge。核心步骤包括：

1.**创建工作空间并克隆 ros-bridge 仓库到 src 目录**

2.**使用 rosdep install 命令安装所有依赖包**

3.**使用 catkin_make 或 catkin build 命令编译工作空间**

## 第四步：启动仿真：连接ROS与自定义地图 ##

1.**终端1 - 启动CARLA服务端：** 在CARLA根目录下运行 ./CarlaUE4.sh 或使用 carla_server 可执行文件。确保服务器启动完成。

2.**终端2 - 启动ROS桥接：** 

首先，设置环境变量。最关键的一步是正确设置 PYTHONPATH，使其包含CARLA的PythonAPI路径（特别是.egg文件），桥接才能与CARLA通信。

然后，使用 roslaunch 启动桥接。如需加载自定义地图，你有两种方法：

*方法一（启动时指定）*：在启动命令中通过 town:='<你的地图名>' 参数直接指定。例如：

>bash
>
>roslaunch carla_ros_bridge carla_ros_bridge.launch town:='Your_Custom_Map'

*方法二（启动后切换）*：先启动桥接连接默认世界，然后通过调用ROS服务（如 /{service_name}）来动态切换到你的自定义地图。

>特别注意：如果CARLA服务端和ROS桥接运行在不同的机器或容器（如本机Windows CARLA，虚拟机内ROS），必须在启动launch文件时额外指定主机IP参数，例如 host:=<主机IP地址>，而不能使用默认的 localhost。

## 版本匹配与常见问题

1.**版本一致性：** CARLA版本、.egg文件版本、carla-ros-bridge分支及ROS版本（Melodic/Noetic）必须互相兼容。

2.**Python环境：** 编译和运行时，Python环境（尤其是Anaconda虚拟环境）可能导致ROS依赖（如rospkg）或CARLA模块导入失败，需仔细检查并安装缺失包。

3.**地图名称：** 确保在ROS桥接中指定的地图名，与地图成功导入CARLA后，在CARLA内部使用的名称完全一致。

# 至此，全部的用roadrunner来高自由度的自主构建地图并导入Carla运行及用ros桥接的全套指南已经结束，开始着手创建自己的地图吧！！！ #
