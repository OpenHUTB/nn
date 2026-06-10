from utils.screen import clear, warn, message, error
from utils.collect import Collector
from utils.piloterror import PilotError
from utils.logger import logger
from src.data import Data, PilotData
from src.model import PilotNet
import carla, random, time, datetime, os

# you could totally enable a feature by which a model trained in a session can be used as fallback if there are no trained models available
# but for this, PilotNet would have to compile and store a model in memory from the start, which may hinder performance of other utilities
# you can do this by uncommenting this line and commenting any lines starting with PilotNet() from this file
# pilotnet = PilotNet(160, 120)

class Menu():

    @staticmethod
    def run_1():
        '使用已生成的数据训练模型'
        logger.info('Starting model training')
        data = Data()
        message('数据加载完成')
        logger.info('Training data loaded successfully')

        message('请输入训练轮数（epochs），默认值为 50')
        epochs = int(input('输入训练轮数 >> ') or 50)
        message('请输入批次大小（batch size），默认值为 64')
        batch_size = int(input('输入批次大小 >> ') or 64)
        message('请输入模型文件名，直接回车使用自动生成的名称')
        name = input('输入名称 >> ') or f"{epochs}epochs_{datetime.datetime.now().strftime('@%Y-%m-%d-%H-%M')}"
        message('请输入图像尺寸，注意较大尺寸会消耗更多内存')
        width = input('输入宽度（默认 160） >> ') or 160
        height = input('输入高度（默认 120） >> ') or 120

        logger.info(f'Training parameters - epochs: {epochs}, batch_size: {batch_size}, model_name: {name}, image_size: {width}x{height}')

        clear()
        try:
            message(f'正在初始化 {width}x{height} 图像的 TensorFlow 模型')
            pilotnet = PilotNet(width, height)
            logger.info('TensorFlow model initialized successfully')
        except Exception as e:
            logger.error(f'Failed to initialize TensorFlow model: {e}')
            raise PilotError('初始化失败，系统可能内存不足，请尝试使用较小的图像尺寸。')
        clear()
        message('开始训练')
        try:
            pilotnet.train(name, data, epochs, steps=None, steps_val=None, batch_size=batch_size)
            logger.info(f'Training completed successfully, model saved as: {name}')
        except Exception as e:
            logger.error(f'Training failed with error: {e}', exc_info=True)
            raise PilotError('训练过程中发生未知错误，请重试。')

    @staticmethod
    def run_2():
        '生成新数据'
        logger.info('Starting data generation')
        message('正在连接到 CARLA 世界')
        client = carla.Client('localhost', 2000)
        try:
            world = client.get_world()
            message('已连接到 CARLA 服务器')
            logger.info('Connected to CARLA server on localhost:2000')
        except Exception as e:
            logger.warning(f'Failed to connect to CARLA on localhost:2000, retrying with WSL address: {e}')
            try:
                warn('CARLA 服务器连接失败，正在尝试使用 WSL 地址重新连接...')
                client = carla.Client('172.17.128.1', 2000)
                world = client.get_world()
                message('已连接到 CARLA 服务器')
                logger.info('Connected to CARLA server on WSL address')
            except Exception as e:
                logger.error(f'Failed to connect to CARLA server: {e}')
                raise PilotError('CARLA 模拟器连接失败。请检查 CARLA 安装，确认模拟器正在端口 2000 上运行。\n如果使用 WSL，请参考故障排除指南。')

        # 地图选择
        available_maps = [
            'Town01',
            'Town02',
            'Town03',
            'Town04',
            'Town05',
            'Town06',
            'Town07',
            'Town10HD'
        ]
        
        message('\n请选择数据采集的地图：')
        for i, map_name in enumerate(available_maps, 1):
            message(f'{i}. {map_name}')
        
        map_choice = int(input('输入选择（1-8，默认 1） >> ') or 1)
        selected_map = available_maps[map_choice - 1]
        logger.info(f'Selected map: {selected_map}')
        
        # 加载选择的地图
        message(f'正在加载地图: {selected_map}...')
        # 增加地图加载超时时间（加载地图可能需要一些时间）
        client.set_timeout(30.0)  # 30秒超时
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                world = client.load_world(selected_map)
                message(f'地图 {selected_map} 加载成功')
                logger.info(f'Map {selected_map} loaded successfully')
                break
            except RuntimeError as e:
                retry_count += 1
                if retry_count < max_retries:
                    warn(f'地图加载超时，正在重试 ({retry_count}/{max_retries})...')
                    logger.warning(f'Map loading timed out, retrying ({retry_count}/{max_retries})')
                else:
                    logger.error(f'Failed to load map {selected_map} after {max_retries} attempts')
                    raise PilotError(f'地图 {selected_map} 加载失败，请确保 CARLA 模拟器正在运行并重试。')

        # 天气选择
        available_weather = [
            ('ClearNoon', '晴朗正午'),
            ('CloudyNoon', '多云正午'),
            ('WetNoon', '雨天正午'),
            ('WetCloudyNoon', '阴雨正午'),
            ('MidRainyNoon', '中雨正午'),
            ('HardRainNoon', '大雨正午'),
            ('SoftRainNoon', '小雨正午'),
            ('ClearSunset', '晴朗日落'),
            ('CloudySunset', '多云日落'),
            ('WetSunset', '雨天日落'),
            ('WetCloudySunset', '阴雨日落'),
            ('MidRainSunset', '中雨日落'),
            ('HardRainSunset', '大雨日落'),
            ('SoftRainSunset', '小雨日落'),
            ('ClearNight', '晴朗夜晚'),
            ('CloudyNight', '多云夜晚'),
            ('WetNight', '雨天夜晚'),
            ('WetCloudyNight', '阴雨夜晚'),
            ('MidRainNight', '中雨夜晚'),
            ('HardRainNight', '大雨夜晚'),
            ('SoftRainNight', '小雨夜晚')
        ]
        
        message('\n请选择天气条件：')
        for i, (weather_id, weather_name) in enumerate(available_weather, 1):
            message(f'{i}. {weather_name}')
        
        weather_choice = int(input('输入选择（1-21，默认 1） >> ') or 1)
        selected_weather_id = available_weather[weather_choice - 1][0]
        selected_weather_name = available_weather[weather_choice - 1][1]
        logger.info(f'Selected weather: {selected_weather_id}')
        
        # Set weather using WeatherParameters constructor
        weather_params = {
            'ClearNoon': carla.WeatherParameters(
                cloudiness=0.0, precipitation=0.0, precipitation_deposits=0.0,
                wind_intensity=0.0, sun_altitude_angle=90.0
            ),
            'CloudyNoon': carla.WeatherParameters(
                cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0,
                wind_intensity=20.0, sun_altitude_angle=90.0
            ),
            'WetNoon': carla.WeatherParameters(
                cloudiness=30.0, precipitation=50.0, precipitation_deposits=30.0,
                wind_intensity=30.0, sun_altitude_angle=90.0
            ),
            'WetCloudyNoon': carla.WeatherParameters(
                cloudiness=80.0, precipitation=60.0, precipitation_deposits=40.0,
                wind_intensity=40.0, sun_altitude_angle=90.0
            ),
            'MidRainyNoon': carla.WeatherParameters(
                cloudiness=80.0, precipitation=80.0, precipitation_deposits=60.0,
                wind_intensity=50.0, sun_altitude_angle=90.0
            ),
            'HardRainNoon': carla.WeatherParameters(
                cloudiness=100.0, precipitation=100.0, precipitation_deposits=80.0,
                wind_intensity=70.0, sun_altitude_angle=90.0
            ),
            'SoftRainNoon': carla.WeatherParameters(
                cloudiness=60.0, precipitation=30.0, precipitation_deposits=20.0,
                wind_intensity=20.0, sun_altitude_angle=90.0
            ),
            'ClearSunset': carla.WeatherParameters(
                cloudiness=0.0, precipitation=0.0, precipitation_deposits=0.0,
                wind_intensity=0.0, sun_altitude_angle=20.0
            ),
            'CloudySunset': carla.WeatherParameters(
                cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0,
                wind_intensity=20.0, sun_altitude_angle=20.0
            ),
            'WetSunset': carla.WeatherParameters(
                cloudiness=30.0, precipitation=50.0, precipitation_deposits=30.0,
                wind_intensity=30.0, sun_altitude_angle=20.0
            ),
            'WetCloudySunset': carla.WeatherParameters(
                cloudiness=80.0, precipitation=60.0, precipitation_deposits=40.0,
                wind_intensity=40.0, sun_altitude_angle=20.0
            ),
            'MidRainSunset': carla.WeatherParameters(
                cloudiness=80.0, precipitation=80.0, precipitation_deposits=60.0,
                wind_intensity=50.0, sun_altitude_angle=20.0
            ),
            'HardRainSunset': carla.WeatherParameters(
                cloudiness=100.0, precipitation=100.0, precipitation_deposits=80.0,
                wind_intensity=70.0, sun_altitude_angle=20.0
            ),
            'SoftRainSunset': carla.WeatherParameters(
                cloudiness=60.0, precipitation=30.0, precipitation_deposits=20.0,
                wind_intensity=20.0, sun_altitude_angle=20.0
            ),
            'ClearNight': carla.WeatherParameters(
                cloudiness=0.0, precipitation=0.0, precipitation_deposits=0.0,
                wind_intensity=0.0, sun_altitude_angle=-90.0
            ),
            'CloudyNight': carla.WeatherParameters(
                cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0,
                wind_intensity=20.0, sun_altitude_angle=-90.0
            ),
            'WetNight': carla.WeatherParameters(
                cloudiness=30.0, precipitation=50.0, precipitation_deposits=30.0,
                wind_intensity=30.0, sun_altitude_angle=-90.0
            ),
            'WetCloudyNight': carla.WeatherParameters(
                cloudiness=80.0, precipitation=60.0, precipitation_deposits=40.0,
                wind_intensity=40.0, sun_altitude_angle=-90.0
            ),
            'MidRainNight': carla.WeatherParameters(
                cloudiness=80.0, precipitation=80.0, precipitation_deposits=60.0,
                wind_intensity=50.0, sun_altitude_angle=-90.0
            ),
            'HardRainNight': carla.WeatherParameters(
                cloudiness=100.0, precipitation=100.0, precipitation_deposits=80.0,
                wind_intensity=70.0, sun_altitude_angle=-90.0
            ),
            'SoftRainNight': carla.WeatherParameters(
                cloudiness=60.0, precipitation=30.0, precipitation_deposits=20.0,
                wind_intensity=20.0, sun_altitude_angle=-90.0
            )
        }
        
        weather = weather_params.get(selected_weather_id, weather_params['ClearNoon'])
        world.set_weather(weather)
        message(f'天气已设置为: {selected_weather_name}')
        logger.info(f'Weather set to: {selected_weather_id}')

        time = int(input('请输入数据采集时间（分钟） >> '))
        logger.info(f'Data generation time: {time} minutes')

        message('是否启用 CARLA 可视化？')
        message('1. 是 - 显示车辆轨迹、控制参数和统计信息')
        message('2. 否 - 仅录制数据，不显示可视化')
        viz_choice = input('输入选择（1-2，默认 1） >> ') or '1'
        enable_visualization = viz_choice == '1'
        logger.info(f'Visualization enabled: {enable_visualization}')

        clear()
        collector = Collector(world, time, enable_visualization=enable_visualization)
        logger.info('Data collection completed')

    @staticmethod
    def run_3():
        '单帧预测'
        import cv2
        import numpy as np
        import datetime
        from PIL import Image, ImageDraw, ImageFont
        
        logger.info('Starting single frame prediction')
        i = 0
        models = []
        message('正在获取已保存的模型列表...')
        with os.scandir('models/') as saved_models:
            for model in saved_models:
                print(f'{i+1}. {model.name}')
                models.append(model.name)
                i+=1
        logger.info(f'Found {len(models)} saved models')

        if len(models) <= 0:
            warn('没有已保存的模型。由于性能问题，当前会话中训练的模型暂不可用。')
            message('请先从菜单训练一个模型，然后重试...')
            logger.warning('No saved models found for prediction')
        else:
            choice = int(input('请选择要使用的模型（输入序号） >> ') or 1)
            while choice < 1 or choice > len(models):
                error('选择错误，请重试...')
                choice = int(input('请选择要使用的模型（输入序号） >> ') or 1)
            model = models[choice - 1]
            message(f'{model} 已选择。')
            logger.info(f'Selected model: {model}')
            path = input('请输入相对于当前目录的图像路径 >> ')
            logger.info(f'Input image path: {path}')
            try:
                frame = PilotData(isTraining=False, path_to=path)
                logger.info('Image loaded successfully')
            except Exception as e:
                logger.error(f'Failed to load image: {e}')
                raise PilotError('加载失败，你输入的路径可能有误，请重新开始...')
            predictions = PilotNet(160, 120, predict=True).predict(frame, given_model=model)
            
            steering = predictions[0][0][0]
            throttle = predictions[1][0][0]
            brake = predictions[2][0][0]
            
            # 获取状态描述
            steering_status = "左拐" if steering < -0.1 else "直行" if abs(steering) <= 0.1 else "右拐"
            throttle_status = "加速" if throttle > 0.5 else "巡航" if throttle > 0.1 else "怠速"
            brake_status = "制动" if brake > 0.3 else "放松"
            
            logger.info(f'Prediction completed - steering: {steering}, throttle: {throttle}, brake: {brake}')
            
            # 加载原始图像用于可视化
            original_img = cv2.imread(path)
            img_height, img_width = original_img.shape[:2]
            
            # 创建可视化图像（放大显示）
            display_img = cv2.resize(original_img, (int(img_width * 1.5), int(img_height * 1.5)))
            display_height, display_width = display_img.shape[:2]
            
            # 转换为 PIL 图像以便显示中文
            display_img_pil = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(display_img_pil)
            
            # 尝试加载中文字体
            try:
                font = ImageFont.truetype("msyh.ttc", 24)  # Windows 系统字体
            except:
                try:
                    font = ImageFont.truetype("simhei.ttf", 24)  # 备用字体
                except:
                    font = ImageFont.load_default()  # 默认字体
            
            font_large = ImageFont.truetype("msyh.ttc", 32) if 'font' in dir() else font
            font_small = ImageFont.truetype("msyh.ttc", 20) if 'font' in dir() else font
            
            # 添加标题（中文）
            draw.text((20, 30), 'PilotNet 预测结果', font=font_large, fill=(0, 255, 0))
            
            # 模型名称显示（添加半透明背景）
            model_text = f'模型: {model[:40]}...' if len(model) > 40 else f'模型: {model}'
            text_bbox = draw.textbbox((20, 75), model_text, font=font_small)
            draw.rectangle([text_bbox[0]-5, text_bbox[1]-3, text_bbox[2]+5, text_bbox[3]+3], fill=(0, 0, 0, 150))
            draw.text((20, 75), model_text, font=font_small, fill=(255, 255, 255))
            
            # 预测结果显示区域（带背景框）
            result_x = 20
            result_y = 120
            spacing = 45
            
            # 转向角度（中文）
            steering_color = (0, 255, 0) if abs(steering) < 0.3 else (0, 255, 255) if abs(steering) < 0.6 else (0, 0, 255)
            steering_text = f'转向角度: {steering:+.3f}  ({steering_status})'
            text_bbox = draw.textbbox((result_x, result_y), steering_text, font=font)
            draw.rectangle([text_bbox[0]-5, text_bbox[1]-3, text_bbox[2]+5, text_bbox[3]+3], fill=(0, 0, 0, 150))
            draw.text((result_x, result_y), steering_text, font=font, fill=steering_color)
            
            # 油门（中文）
            throttle_color = (255, 0, 0) if throttle > 0.5 else (0, 255, 0)
            throttle_text = f'油门压力: {throttle:.3f}  ({throttle_status})'
            text_bbox = draw.textbbox((result_x, result_y + spacing), throttle_text, font=font)
            draw.rectangle([text_bbox[0]-5, text_bbox[1]-3, text_bbox[2]+5, text_bbox[3]+3], fill=(0, 0, 0, 150))
            draw.text((result_x, result_y + spacing), throttle_text, font=font, fill=throttle_color)
            
            # 刹车（中文）
            brake_color = (0, 0, 255) if brake > 0.5 else (0, 255, 0)
            brake_text = f'刹车压力: {brake:.3f}  ({brake_status})'
            text_bbox = draw.textbbox((result_x, result_y + spacing * 2), brake_text, font=font)
            draw.rectangle([text_bbox[0]-5, text_bbox[1]-3, text_bbox[2]+5, text_bbox[3]+3], fill=(0, 0, 0, 150))
            draw.text((result_x, result_y + spacing * 2), brake_text, font=font, fill=brake_color)
            
            # 转换回 OpenCV 格式
            display_img = cv2.cvtColor(np.array(display_img_pil), cv2.COLOR_RGB2BGR)
            
            # 绘制方向盘可视化
            steering_wheel_radius = 60
            steering_center_x = display_width - 100
            steering_center_y = display_height - 100
            
            # 方向盘外圈
            cv2.circle(display_img, (steering_center_x, steering_center_y), steering_wheel_radius, (200, 200, 200), 3)
            cv2.circle(display_img, (steering_center_x, steering_center_y), steering_wheel_radius - 10, (100, 100, 100), 2)
            
            # 根据转向角度旋转方向盘
            rotation_angle = -steering * 45  # 将转向角度转换为方向盘角度（最大45度）
            radians = np.radians(rotation_angle)
            
            # 方向盘辐条
            for i in range(3):
                angle = np.radians(i * 120) + radians
                x1 = steering_center_x + int(np.cos(angle) * (steering_wheel_radius - 15))
                y1 = steering_center_y + int(np.sin(angle) * (steering_wheel_radius - 15))
                x2 = steering_center_x - int(np.cos(angle) * (steering_wheel_radius - 15))
                y2 = steering_center_y - int(np.sin(angle) * (steering_wheel_radius - 15))
                cv2.line(display_img, (x1, y1), (x2, y2), (200, 200, 200), 3)
            
            # 方向盘中心
            cv2.circle(display_img, (steering_center_x, steering_center_y), 8, (255, 0, 0), -1)
            
            # 转向指示器
            indicator_length = 30
            indicator_x = steering_center_x + int(np.cos(radians) * indicator_length)
            indicator_y = steering_center_y + int(np.sin(radians) * indicator_length)
            cv2.line(display_img, (steering_center_x, steering_center_y), (indicator_x, indicator_y), (0, 255, 0), 3)
            cv2.circle(display_img, (indicator_x, indicator_y), 5, (0, 255, 0), -1)
            
            # 状态指示条 - 使用 PIL 绘制以支持中文
            display_img_pil = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(display_img_pil)
            
            status_bar_y = display_height - 40
            
            # 绘制状态栏背景
            draw.rectangle([(20, status_bar_y), (display_width - 140, status_bar_y + 25)], fill=(50, 50, 50))
            
            # 油门指示条
            throttle_width = int(throttle * 150)
            draw.rectangle([(30, status_bar_y + 5), (30 + throttle_width, status_bar_y + 20)], fill=(0, 255, 0))
            draw.text((30, status_bar_y - 8), '油门', font=font_small, fill=(0, 255, 0))
            
            # 刹车指示条
            brake_width = int(brake * 150)
            draw.rectangle([(200, status_bar_y + 5), (200 + brake_width, status_bar_y + 20)], fill=(0, 0, 255))
            draw.text((200, status_bar_y - 8), '刹车', font=font_small, fill=(0, 0, 255))
            
            # 转换回 OpenCV 格式
            display_img = cv2.cvtColor(np.array(display_img_pil), cv2.COLOR_RGB2BGR)
            
            # 保存预测结果图像
            save_dir = 'predictions/'
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            save_path = f'{save_dir}prediction_{timestamp}.png'
            cv2.imwrite(save_path, display_img)
            logger.info(f'Prediction result saved to: {save_path}')
            
            # 显示可视化结果
            cv2.imshow('PilotNet Prediction', display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # 终端显示详细结果
            clear()
            message('='*50)
            message('          PilotNet 单帧预测结果')
            message('='*50)
            message(f'\n模型名称: {model}')
            message(f'输入图像: {path}')
            message(f'预测时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            message('\n' + '-'*50)
            message('预测结果:')
            message(f'  转向角度: {steering:+.3f}  ({("左拐" if steering < -0.1 else "直行" if abs(steering) <= 0.1 else "右拐")})')
            message(f'  油门压力: {throttle:.3f}  ({("加速" if throttle > 0.3 else "巡航" if throttle > 0.1 else "怠速")})')
            message(f'  刹车压力: {brake:.3f}  ({("制动" if brake > 0.3 else "放松")})')
            message('\n' + '-'*50)
            message(f'结果已保存至: {save_path}')
            message('='*50)
            input('\n按 [ENTER] 继续...')

    @staticmethod
    def run_4():
        '实时视频预测'
        logger.warning('Live video prediction feature requested but not yet implemented')
        raise PilotError('抱歉，实时视频预测功能尚未实现，正在开发中，请耐心等待。')

    @staticmethod
    def run_5():
        '退出程序'
        logger.info('User requested to exit the application')
        message('感谢使用 PilotNet，如有问题请在 GitHub 上报告。')

    @staticmethod
    def execute(user_input):
        task_name = f'run_{user_input}'
        try:
            menu = getattr(Menu, task_name)
            clear()
        except AttributeError:
            error_messages = [
                '输入无效，请查看菜单选项...',
                '这个选项不存在，请重试...',
                '抱歉，这个选项不在菜单中...',
                '我无法理解你的选择，请重新输入...',
                '请输入正确的选项序号...']
            raise PilotError(random.choice(error_messages))
        else:
            menu()

    @staticmethod
    def generate_instructions():
        do_methods = [m for m in dir(Menu) if m.startswith('run_')]
        menu_string = "\n".join(
            [f'{method[-1]}.  {getattr(Menu, method).__doc__}' for method in do_methods])
        print(menu_string)

    @staticmethod
    def run():
        user_input = 0
        while(user_input != 5):
            clear()
            Menu.generate_instructions()
            user_input = int(input("请输入你的选择 >> "))
            try:
                Menu.execute(user_input)
            except PilotError:
                input('按 [ENTER] 继续')
            except KeyboardInterrupt:
                message('感谢使用 PilotNet，如有问题请在 GitHub 上报告。')

def main():
    logger.info('PilotNet application started')
    logger.info('Log file: %s', logger.get_log_path())
    try:
        Menu.run()
    except Exception as e:
        logger.critical(f'Application crashed with error: {e}', exc_info=True)
        raise

if __name__ == '__main__':
    main()
