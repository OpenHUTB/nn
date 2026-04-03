# for Python3-based torcs python robot client
import socket
import sys
import getopt
import os
import time
PI= 3.14159265359

data_size = 2**17

# 初始化帮助信息
ophelp=  'Options:\n'
ophelp+= ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp+= ' --port, -p <port>    TORCS port. [3001]\n'
ophelp+= ' --id, -i <id>        ID for server. [SCR]\n'
ophelp+= ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp+= ' --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp+= ' --track, -t <track>  Your name for this track. Used for learning. [unknown]\n'
ophelp+= ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp+= ' --debug, -d          Output full telemetry.\n'
ophelp+= ' --help, -h           Show this help.\n'
ophelp+= ' --version, -v        Show current version.'
usage= 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage= usage + ophelp
version= "20130505-2"

def clip(v,lo,hi):
    if v<lo: return lo
    elif v>hi: return hi
    else: return v

def bargraph(x,mn,mx,w,c='X'):
    """绘制简单的ASCII条形图，便于可视化数据
    参数:
        x: 传感器数值
        mn: 绘图最小值
        mx: 绘图最大值
        w: 图形宽度（字符数）
        c: 绘图使用的字符
    返回:
        ASCII条形图字符串
    """
    if not w: return '' # 无宽度时返回空
    if x<mn: x= mn      # 限制下限
    if x>mx: x= mx      # 限制上限
    tx= mx-mn # 可显示的数值范围
    if tx<=0: return 'backwards' # 无效范围
    upw= tx/float(w) # 每个字符代表的数值
    if upw<=0: return 'what?' # 防止除零错误
    negpu, pospu, negnonpu, posnonpu= 0,0,0,0
    if mn < 0:  # 包含负数部分
        if x < 0:  # 数值在负数区域
            negpu = -x + min(0, mx)
            negnonpu = -mn + x
        else:  # 数值在正数区域，负数部分为空
            negnonpu = -mn + min(0, mx)
    if mx > 0:  # 包含正数部分
        if x > 0:  # 数值在正数区域
            pospu = x - max(0, mn)
            posnonpu = mx - x
        else:  # 数值在负数区域，正数部分为空
            posnonpu = mx - max(0, mn)

    # 构建条形图字符串
    nnc= int(negnonpu/upw)*'-'
    npc= int(negpu/upw)*c
    ppc= int(pospu/upw)*c
    pnc= int(posnonpu/upw)*'_'
    return '[%s]' % (nnc+npc+ppc+pnc)

class Client():
    def __init__(self,H=None,p=None,i=None,e=None,t=None,s=None,d=None,vision=False):
        # 可在此修改默认选项值
        self.vision = vision

        # 默认配置
        self.host = 'localhost'  # 服务器地址
        self.port = 3001  # 服务器端口
        self.sid = 'SCR'  # 客户端ID
        self.maxEpisodes = 1  # 最大学习回合数
        self.trackname = 'unknown'  # 赛道名称
        self.stage = 3  # 比赛阶段
        self.debug = False  # 调试模式
        self.maxSteps = 100000  # 最大仿真步数（50步/秒）

        self.parse_the_command_line()  # 解析命令行参数

        # 覆盖默认配置
        if H: self.host= H
        if p: self.port= p
        if i: self.sid= i
        if e: self.maxEpisodes= e
        if t: self.trackname= t
        if s: self.stage= s
        if d: self.debug= d

        self.S = ServerState()  # 服务器状态对象
        self.R = DriverAction()  # 驱动动作对象
        self.setup_connection()  # 建立服务器连接

    def setup_connection(self):
        # == 创建UDP套接字 ==
        try:
            self.so= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as emsg:
            print('Error: Could not create socket...')
            sys.exit(-1)
        # == 初始化服务器连接 ==
        self.so.settimeout(1)

        n_fail = 5
        while True:
            # 配置赛道传感器角度，可自定义
            # 原始配置: "-90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90"
            # 优化后的配置，更紧凑的角度分布
            a= "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"

            initmsg='%s(init %s)' % (self.sid,a)

            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error as emsg:
                sys.exit(-1)
            sockdata= str()
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print("Waiting for server on %d............" % self.port)
                print("Count Down : " + str(n_fail))
                if n_fail < 0:
                    print("relaunch torcs")
                    os.system('pkill torcs')
                    time.sleep(1.0)
                    if self.vision is False:
                        os.system('torcs -nofuel -nodamage -nolaptime &')
                    else:
                        os.system('torcs -nofuel -nodamage -nolaptime -vision &')

                    time.sleep(1.0)
                    os.system('sh autostart.sh')
                    n_fail = 5
                n_fail -= 1

            identify = '***identified***'
            if identify in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def parse_the_command_line(self):
        try:
            (opts, args) = getopt.getopt(sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                       ['host=','port=','id=','steps=',
                        'episodes=','track=','stage=',
                        'debug','help','version'])
        except getopt.error as why:
            print('getopt error: %s\n%s' % (why, usage))
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] == '-h' or opt[0] == '--help':
                    print(usage)
                    sys.exit(0)
                if opt[0] == '-d' or opt[0] == '--debug':
                    self.debug= True
                if opt[0] == '-H' or opt[0] == '--host':
                    self.host= opt[1]
                if opt[0] == '-i' or opt[0] == '--id':
                    self.sid= opt[1]
                if opt[0] == '-t' or opt[0] == '--track':
                    self.trackname= opt[1]
                if opt[0] == '-s' or opt[0] == '--stage':
                    self.stage= int(opt[1])
                if opt[0] == '-p' or opt[0] == '--port':
                    self.port= int(opt[1])
                if opt[0] == '-e' or opt[0] == '--episodes':
                    self.maxEpisodes= int(opt[1])
                if opt[0] == '-m' or opt[0] == '--steps':
                    self.maxSteps= int(opt[1])
                if opt[0] == '-v' or opt[0] == '--version':
                    print('%s %s' % (sys.argv[0], version))
                    sys.exit(0)
        except ValueError as why:
            print('Bad parameter \'%s\' for option %s: %s\n%s' % (
                                       opt[1], opt[0], why, usage))
            sys.exit(-1)
        if len(args) > 0:
            print('Superflous input? %s\n%s' % (', '.join(args), usage))
            sys.exit(-1)

    def get_servers_input(self):
        """接收并解析服务器发送的状态数据
        服务器输入将存储在ServerState对象中
        """
        if not self.so: return
        sockdata= str()

        while True:
            try:
                # 接收服务器数据
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print('.', end=' ')
                #print "Waiting for data on %d.............." % self.port

            # 处理不同类型的服务器消息
            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                continue
            elif '***shutdown***' in sockdata:
                print((("Server has stopped the race on %d. "+
                        "You were in %d place.") %
                        (self.port,self.S.d['racePos'])))
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                print("Server has restarted the race on %d." % self.port)
                self.shutdown()
                return
            elif not sockdata: # 空数据
                continue       # 重新接收
            else:
                # 解析服务器数据
                self.S.parse_server_str(sockdata)
                if self.debug:
                    sys.stderr.write("\x1b[2J\x1b[H")  # 清屏
                    print(self.S)
                break  # 解析完成，退出循环

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1],str(emsg[0])))
            sys.exit(-1)
        if self.debug: print(self.R.fancyout())
        # 如需简单输出，可使用：
        # if self.debug: print(self.R)

    def shutdown(self):
        if not self.so: return
        print(("Race terminated or %d steps elapsed. Shutting down %d."
               % (self.maxSteps,self.port)))
        self.so.close()
        self.so = None
        #sys.exit() # 无需强制退出

class ServerState():
    '''What the server is reporting right now.'''
    def __init__(self):
        self.servstr= str()
        self.d= dict()

    def parse_server_str(self, server_string):
        '''Parse the server string.'''
        self.servstr= server_string.strip()[:-1]
        sslisted= self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w= i.split(' ')
            self.d[w[0]]= destringify(w[1:])

    def __repr__(self):
        # 注释下一行可输出原始数据
        return self.fancyout()
        # -------------------------------------
        out= str()
        for k in sorted(self.d):
            strout= str(self.d[k])
            if type(self.d[k]) is list:
                strlist= [str(i) for i in self.d[k]]
                strout= ', '.join(strlist)
            out+= "%s: %s\n" % (k,strout)
        return out

    def fancyout(self):
        """格式化输出服务器状态信息（便于调试）"""
        out= str()
        # 选择要显示的传感器（按显示顺序）
        sensors = [
            # 'curLapTime',    # 当前圈时间
            # 'lastLapTime',   # 上一圈时间
            'stucktimer',  # 卡滞计时器
            # 'damage',        # 损伤值
            # 'focus',         # 焦点
            'fuel',  # 燃油量
            # 'gear',          # 档位
            'distRaced',  # 已行驶距离
            'distFromStart',  # 距起点距离
            # 'racePos',       # 比赛名次
            'opponents',  # 对手位置
            'wheelSpinVel',  # 车轮转速
            'z',  # Z轴位置
            'speedZ',  # Z方向速度
            'speedY',  # Y方向速度
            'speedX',  # X方向速度（前进/后退）
            'targetSpeed',  # 目标速度
            'rpm',  # 发动机转速
            'skid',  # 侧滑程度
            'slip',  # 打滑程度
            'track',  # 赛道传感器
            'trackPos',  # 赛道位置（-1=最左，1=最右）
            'angle',  # 车身角度
        ]

        #for k in sorted(self.d): # 显示所有传感器
        for k in sensors:
            if type(self.d.get(k)) is list:  # 处理列表类型数据
                if k == 'track':  # 赛道传感器的友好显示
                    strout= str()
                 #  for tsensor in self.d['track']:
                 #      if   tsensor >180: oc= '|'
                 #      elif tsensor > 80: oc= ';'
                 #      elif tsensor > 60: oc= ','
                 #      elif tsensor > 39: oc= '.'
                 #      #elif tsensor > 13: oc= chr(int(tsensor)+65-13)
                 #      elif tsensor > 13: oc= chr(int(tsensor)+97-13)
                 #      elif tsensor >  3: oc= chr(int(tsensor)+48-3)
                 #      else: oc= '_'
                 #      strout+= oc
                 #  strout= ' -> '+strout[:9] +' ' + strout[9] + ' ' + strout[10:]+' <-'
                    raw_tsens= ['%.1f'%x for x in self.d['track']]
                    strout+= ' '.join(raw_tsens[:9])+'_'+raw_tsens[9]+'_'+' '.join(raw_tsens[10:])
                elif k == 'opponents': # 对手传感器的友好显示
                    strout= str()
                    for osensor in self.d['opponents']:
                        if   osensor >190: oc= '_'
                        elif osensor > 90: oc= '.'
                        elif osensor > 39: oc= chr(int(osensor/2)+97-19)
                        elif osensor > 13: oc= chr(int(osensor)+65-13)
                        elif osensor >  3: oc= chr(int(osensor)+48-3)
                        else: oc= '?'
                        strout+= oc
                    strout= ' -> '+strout[:18] + ' ' + strout[18:]+' <-'
                else:
                    strlist= [str(i) for i in self.d[k]]
                    strout= ', '.join(strlist)
                    else:  # 非列表类型数据
                    if k == 'gear':  # 档位可视化（与RPM显示重复）
                        gs = '_._._._._._._._._'
                        p = int(self.d['gear']) * 2 + 2  # 位置
                        l = '%d' % self.d['gear']  # 标签
                        if l == '-1':
                            l = 'R'  # 倒挡
                        if l == '0':
                            l = 'N'  # 空挡
                        strout = gs[:p] + '(%s)' % l + gs[p + 3:]
                    elif k == 'damage':
                        # 损伤值+条形图
                        strout = '%6.0f %s' % (self.d[k], bargraph(self.d[k], 0, 10000, 50, '~'))
                    elif k == 'fuel':
                        # 燃油量+条形图
                        strout = '%6.0f %s' % (self.d[k], bargraph(self.d[k], 0, 100, 50, 'f'))
                    elif k == 'speedX':
                        # X方向速度（前进/后退）+条形图
                        cx = 'X'
                        if self.d[k] < 0:
                            cx = 'R'  # 后退
                        strout = '%6.1f %s' % (self.d[k], bargraph(self.d[k], -30, 300, 50, cx))
                    elif k == 'speedY':
                        # Y方向速度（横向）+条形图（反转显示更直观）
                        strout = '%6.1f %s' % (self.d[k], bargraph(self.d[k] * -1, -25, 25, 50, 'Y'))
                    elif k == 'speedZ':
                        # Z方向速度+条形图
                        strout = '%6.1f %s' % (self.d[k], bargraph(self.d[k], -13, 13, 50, 'Z'))
                    elif k == 'z':
                        # Z轴位置+条形图
                        strout = '%6.3f %s' % (self.d[k], bargraph(self.d[k], .3, .5, 50, 'z'))
                    elif k == 'trackPos':
                        # 赛道位置+条形图（反转显示更直观）
                        cx = '<'
                        if self.d[k] < 0:
                            cx = '>'
                        strout = '%6.3f %s' % (self.d[k], bargraph(self.d[k] * -1, -1, 1, 50, cx))
                    elif k == 'stucktimer':
                        # 卡滞计时器
                        if self.d[k]:
                            strout = '%3d %s' % (self.d[k], bargraph(self.d[k], 0, 300, 50, "'"))
                        else:
                            strout = '未卡滞!'
                    elif k == 'rpm':
                        # 发动机转速+档位显示
                        g = self.d['gear']
                        if g < 0:
                            g = 'R'
                        else:
                            g = '%1d' % g
                        strout = bargraph(self.d[k], 0, 10000, 50, g)
                    elif k == 'angle':
                        # 车身角度可视化
                        asyms = [
                            "  !  ", ".|'  ", "./'  ", "_.-  ", ".--  ", "..-  ",
                            "---  ", ".__  ", "-._  ", "'-.  ", "'\.  ", "'|.  ",
                            "  |  ", "  .|'", "  ./'", "  .-'", "  _.-", "  __.",
                            "  ---", "  --.", "  -._", "  -..", "  '\.", "  '|."]
                        rad = self.d[k]
                        deg = int(rad * 180 / PI)
                        symno = int(.5 + (rad + PI) / (PI / 12))
                        symno = symno % (len(asyms) - 1)
                        strout = '%5.2f %3d (%s)' % (rad, deg, asyms[symno])
                    elif k == 'skid':
                        # 侧滑程度（基于车轮转速计算）
                        frontwheelradpersec = self.d['wheelSpinVel'][0]
                        skid = 0
                        if frontwheelradpersec:
                            skid = .5555555555 * self.d['speedX'] / frontwheelradpersec - .66124
                        strout = bargraph(skid, -.05, .4, 50, '*')
                    elif k == 'slip':
                        # 打滑程度（基于车轮转速差计算）
                        frontwheelradpersec = self.d['wheelSpinVel'][0]
                        slip = 0
                        if frontwheelradpersec:
                            slip = ((self.d['wheelSpinVel'][2] + self.d['wheelSpinVel'][3]) -
                                    (self.d['wheelSpinVel'][0] + self.d['wheelSpinVel'][1]))
                        strout = bargraph(slip, -5, 150, 50, '@')
                    else:
                        strout = str(self.d[k])
                out += "%s: %s\n" % (k, strout)
                return out


class DriverAction():
    '''存储驾驶员要发送给服务器的动作指令
    生成的指令格式示例：
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus 0)(meta 0) 或
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus -90 -45 0 45 90)(meta 0)
    '''

    def __init__(self):
        self.actionstr = str()
        # "d" 是动作数据字典
        self.d = {
            'accel': 0.2,  # 油门 (0-1)
            'brake': 0,  # 刹车 (0-1)
            'clutch': 0,  # 离合 (0-1)
            'gear': 1,  # 档位 (-1=倒挡, 0=空挡, 1-6=前进挡)
            'steer': 0,  # 转向 (-1=左满舵, 1=右满舵)
            'focus': [-90, -45, 0, 45, 90],  # 焦点角度
            'meta': 0  # 元信息 (0/1)
        }

    def clip_to_limits(self):
        """将动作参数限制在有效范围内
        服务器不接受超出范围的参数（如steer=9483.323），
        因此默认对所有参数进行裁剪。如需特殊限制，可使用clip函数
        """
        self.d['steer'] = clip(self.d['steer'], -1, 1)  # 转向限制在[-1,1]
        self.d['brake'] = clip(self.d['brake'], 0, 1)  # 刹车限制在[0,1]
        self.d['accel'] = clip(self.d['accel'], 0, 1)  # 油门限制在[0,1]
        self.d['clutch'] = clip(self.d['clutch'], 0, 1)  # 离合限制在[0,1]

        # 档位限制
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear'] = 0

        # 元信息限制
        if self.d['meta'] not in [0, 1]:
            self.d['meta'] = 0

        # 焦点角度限制
        if type(self.d['focus']) is not list or min(self.d['focus']) < -180 or max(self.d['focus']) > 180:
            self.d['focus'] = 0

    def __repr__(self):
        """生成服务器可识别的动作字符串"""
        self.clip_to_limits()  # 先裁剪参数范围
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if not type(v) is list:
                out += '%.3f' % v
            else:
                out += ' '.join([str(x) for x in v])
            out += ')'
        return out

    def fancyout(self):
        '''格式化输出驾驶员动作指令（便于调试）'''
        out = str()
        od = self.d.copy()
        od.pop('gear', '')  # 暂不显示档位
        od.pop('meta', '')  # 暂不显示元信息
        od.pop('focus', '')  # 暂不显示焦点

        for k in sorted(od):
            if k == 'clutch' or k == 'brake' or k == 'accel':
                # 离合/刹车/油门 + 条形图
                strout = '%6.3f %s' % (od[k], bargraph(od[k], 0, 1, 50, k[0].upper()))
            elif k == 'steer':
                # 转向 + 条形图（反转显示更直观）
                strout = '%6.3f %s' % (od[k], bargraph(od[k] * -1, -1, 1, 50, 'S'))
            else:
                strout = str(od[k])
            out += "%s: %s\n" % (k, strout)
        return out


# == 通用工具函数 ==
def destringify(s):
    '''将字符串转换为数值，或字符串列表转换为数值列表
    参数:
        s: 字符串或字符串列表
    返回:
        转换后的数值/数值列表（转换失败则返回原字符串）
    '''
    if not s:
        return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            print("无法从 %s 中提取数值" % s)
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]


def drive_example(c):
    '''示例驱动函数（仅作参考）
    该函数能完成基本的赛道行驶，但建议编写自定义的`drive()`函数
    参数:
        c: Client客户端对象
    '''
    S, R = c.S.d, c.R.d
    target_speed = 100  # 目标速度

    # 转向控制：向弯道中心转向
    R['steer'] = S['angle'] * 10 / PI
    # 转向控制：向赛道中心修正
    R['steer'] -= S['trackPos'] * .10

    # 油门控制
    if S['speedX'] < target_speed - (R['steer'] * 50):
        R['accel'] += .01  # 加速
    else:
        R['accel'] -= .01  # 减速
    if S['speedX'] < 10:
        R['accel'] += 1 / (S['speedX'] + .1)  # 低速时快速加速

    # 牵引力控制系统
    if ((S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) -
            (S['wheelSpinVel'][0] + S['wheelSpinVel'][1]) > 5):
        R['accel'] -= .2  # 检测到打滑，降低油门

    # 自动变速箱
    R['gear'] = 1
    if S['speedX'] > 50:
        R['gear'] = 2
    if S['speedX'] > 80:
        R['gear'] = 3
    if S['speedX'] > 110:
        R['gear'] = 4
    if S['speedX'] > 140:
        R['gear'] = 5
    if S['speedX'] > 170:
        R['gear'] = 6
    return

# ================ MAIN ================
if __name__ == "__main__":
    C= Client(p=3101)
    for step in range(C.maxSteps,0,-1):
        C.get_servers_input()
        drive_example(C)
        C.respond_to_server()
    C.shutdown()
