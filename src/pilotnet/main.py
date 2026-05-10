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
            choice = int(input('请选择要使用的模型（输入序号） >> ') or 0)
            while choice not in models:
                try:
                    choice = models[choice-1]
                    message(f'{choice} 已选择。')
                    logger.info(f'Selected model: {choice}')
                except:
                    error('选择错误，请重试...')
            model = choice
            path = input('请输入相对于当前目录的图像路径 >> ')
            logger.info(f'Input image path: {path}')
            try:
                frame = PilotData(isTraining=False, path_to=path)
                logger.info('Image loaded successfully')
            except Exception as e:
                logger.error(f'Failed to load image: {e}')
                raise PilotError('加载失败，你输入的路径可能有误，请重新开始...')
            predictions = PilotNet(160, 120, predict=True).predict(frame, given_model=model)
            logger.info(f'Prediction completed - steering: {predictions[0][0][0]}, throttle: {predictions[1][0][0]}, brake: {predictions[2][0][0]}')
            clear()
            message('预测结果：')
            message(f'转向角度: {predictions[0][0][0]}')
            message(f'油门: {predictions[1][0][0]}')
            message(f'刹车: {predictions[2][0][0]}')
            input('按 [ENTER] 继续...')

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
