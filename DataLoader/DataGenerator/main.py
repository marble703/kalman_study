import math
import os
from datetime import datetime
from tqdm import tqdm

def wrap_angle(angle):
    """将角度限制到 [-pi, pi) 之间"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def generate_motion(waypoints, speed, sample_rate, sine_enabled=False, sine_amplitude=0.0, sine_frequency=0.0, sub_ang_speed=0.0):
    """
    生成运动状态序列。
    :param waypoints: [(x,y,z), ...]，一组三维路径点
    :param speed: 运动速度（单位与坐标一致，如米/秒）
    :param sample_rate: 采样率(Hz)
    :return: [
        {'t': 时间（秒）, 'pos': (x,y,z), 'vel': (vx,vy,vz)},
        ...
    ]
    """
    states = []
    t = 0.0
    dt = 1.0 / sample_rate

    for i in range(len(waypoints) - 1):
        wp0 = waypoints[i]; wp1 = waypoints[i+1]
        x0, y0, z0 = wp0['pos']; yaw0, roll0, pitch0 = wp0['yaw'], wp0['roll'], wp0['pitch']
        x1, y1, z1 = wp1['pos']; yaw1, roll1, pitch1 = wp1['yaw'], wp1['roll'], wp1['pitch']
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist == 0:
            continue
        # 单位方向向量
        ux, uy, uz = dx/dist, dy/dist, dz/dist
        segment_time = dist / speed
        steps = int(math.ceil(segment_time * sample_rate))
        for step in range(steps):
            # 当前时间偏移
            tt = step * dt
            # 插值计算目标朝向
            yaw_t = yaw0 + (yaw1 - yaw0) * (tt / segment_time)
            yaw_t = wrap_angle(yaw_t)
            roll_t = roll0 + (roll1 - roll0) * (tt / segment_time)
            pitch_t = pitch0 + (pitch1 - pitch0) * (tt / segment_time)
            # 防止最后一步越界
            travel = min(speed * tt, dist)
            px = x0 + ux * travel
            py = y0 + uy * travel
            pz = z0 + uz * travel
            # 主目标无竖直正弦运动
            pos = (px, py, pz)
            vel = (ux * speed, uy * speed, uz * speed)
            # 添加子角速度等，可在写出时使用pos和yaw_t
            states.append({'t': t + tt, 'pos': pos, 'vel': vel, 'yaw': yaw_t, 'roll': roll_t, 'pitch': pitch_t})
        t += segment_time

    # 确保包含最后一点
    wpf = waypoints[-1]
    xf, yf, zf = wpf['pos']; yaw_f, roll_f, pitch_f = wpf['yaw'], wpf['roll'], wpf['pitch']
    # 末点主目标保持静止，无正弦运动
    pos = (xf, yf, zf)
    vel = (0.0, 0.0, 0.0)
    yaw_f = wrap_angle(yaw_f)
    states.append({'t': t, 'pos': pos, 'vel': vel, 'yaw': yaw_f, 'roll': roll_f, 'pitch': pitch_f})
    return states

if __name__ == '__main__':
    # 运动参数配置
    # 运动路径点配置
    speed = 1
    sample_rate = 200
    idle_duration = 10  # 秒
    # 可选竖直正弦运动配置
    sine_enabled = False   # 启用竖直正弦运动
    sine_amplitude = 0.1  # 振幅
    sine_frequency = 0.5  # 频率(Hz)

    # 子目标配置
    sub_enabled = True      # 启用子目标
    sub_num = 4             # 子目标数量，可设为3或4
    # 对于3个目标，使用单圆半径
    sub_radius = 0.533
    # 对于4个目标，使用两个同心圆参数
    sub_r1 = 0.4
    sub_dz1 = 0.2
    sub_r2 = 0.6
    sub_dz2 = 0.3
    # 子目标绕目标yaw轴旋转速度 (rad/s)
    sub_ang_speed = 1

    # 示例用法：从文件读取点，或直接写列表
    # waypoints = [(0,0,0),(1,2,0),(3,2,1)]
    waypoints = []
    with open('points.txt') as f:
        for line in f:
            vals = list(map(float, line.split()))
            if len(vals) == 3:
                x, y, z = vals; yaw = roll = pitch = 0.0
            elif len(vals) == 6:
                x, y, z, yaw, roll, pitch = vals
            else:
                continue
            waypoints.append({'pos': (x, y, z), 'yaw': yaw, 'roll': roll, 'pitch': pitch})

    trajectory = generate_motion(
        waypoints, speed, sample_rate,
        sine_enabled, sine_amplitude, sine_frequency,
        sub_ang_speed
    )
    # 添加起始空闲时间：主目标静止，子目标继续旋转和正弦运动

    if idle_duration > 0:
        dt = 1.0 / sample_rate
        # 初始位置和姿态
        wp0 = waypoints[0]
        x0, y0, z0 = wp0['pos']
        yaw0, roll0, pitch0 = wp0['yaw'], wp0['roll'], wp0['pitch']
        idle_states = []
        steps_idle = int(idle_duration * sample_rate)
        for step in range(steps_idle):
            t_id = step * dt
            # 正弦偏移
            if sine_enabled:
                off_z = sine_amplitude * math.sin(2 * math.pi * sine_frequency * t_id)
                off_vz = sine_amplitude * 2 * math.pi * sine_frequency * math.cos(2 * math.pi * sine_frequency * t_id)
            else:
                off_z = off_vz = 0.0
            idle_states.append({
                't': t_id,
                'pos': (x0, y0, z0 + off_z),
                'vel': (0.0, 0.0, off_vz),
                'yaw': wrap_angle(yaw0),
                'roll': roll0,
                'pitch': pitch0
            })
        # 迁移后续轨迹时间
        for st in trajectory:
            st['t'] += idle_duration
        # 合并空闲和运动轨迹
        trajectory = idle_states + trajectory

    # 检查并创建输出目录，生成文件路径
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, 'trajectory.txt')
    if os.path.exists(filename):
        new_name = os.path.join(output_dir, f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        print(f"文件 {filename} 已存在，将创建新文件: {new_name}")
        filename = new_name

    # 将输出保存到文件，显示进度条
    with open(filename, 'w') as f:
        # 写入表头，包含主目标及其姿态和子目标位置与姿态
        header = ['t','x','y','z','vx','vy','vz','yaw','roll','pitch']
        if sub_enabled:
            count = sub_num
            for i in range(count):
                # 每个子目标输出位置(x,y,z)和姿态(yaw, roll, pitch)
                header += [f'sx{i+1}', f'sy{i+1}', f'sz{i+1}', f'syaw{i+1}', f'sroll{i+1}', f'spitch{i+1}']
        f.write(','.join(header) + '\n')
        for state in tqdm(trajectory, desc='Writing trajectory'):
             t0 = state['t']
             x0, y0, z0 = state['pos']
             vx, vy, vz = state['vel']
             yaw_t = state.get('yaw', 0.0)
             roll_t = state.get('roll', 0.0)
             pitch_t = state.get('pitch', 0.0)
            # 子目标竖直正弦偏移 (只对子目标生效)
             if sine_enabled:
                 sub_sine_z = sine_amplitude * math.sin(2 * math.pi * sine_frequency * t0)
             else:
                 sub_sine_z = 0.0
            # 子目标相对主目标的俯仰偏移（3个时-15°，4个时+15°）
             pitch_offset = -math.radians(15) if sub_num == 3 else math.radians(15)
             sub_coords = []
             if sub_enabled:
                 if sub_num == 3:
                     for idx in range(3):
                         base_angle = 2 * math.pi / 3 * idx
                         ang = wrap_angle(base_angle + yaw_t + sub_ang_speed * t0)
                         sx = x0 + sub_radius * math.cos(ang)
                         sy = y0 + sub_radius * math.sin(ang)
                         sz = z0 + sub_sine_z
                         # 子目标姿态：yaw=ang, roll=主目标roll, pitch=主目标pitch+偏移
                         sub_coords.extend([sx, sy, sz, ang, roll_t, pitch_t + pitch_offset])
                 elif sub_num == 4:
                     for base_angle in (0, math.pi):
                         ang = wrap_angle(base_angle + yaw_t + sub_ang_speed * t0)
                         sx = x0 + sub_r1 * math.cos(ang)
                         sy = y0 + sub_r1 * math.sin(ang)
                         sz = z0 + sub_dz1 + sub_sine_z
                         sub_coords.extend([sx, sy, sz, ang, roll_t, pitch_t + pitch_offset])
                     for base_angle in (math.pi/2, 3 * math.pi/2):
                         ang = wrap_angle(base_angle + yaw_t + sub_ang_speed * t0)
                         sx = x0 + sub_r2 * math.cos(ang)
                         sy = y0 + sub_r2 * math.sin(ang)
                         sz = z0 + sub_dz2 + sub_sine_z
                         sub_coords.extend([sx, sy, sz, ang, roll_t, pitch_t + pitch_offset])
            # 汇总所有数值并写入
             values = [t0, x0, y0, z0, vx, vy, vz, yaw_t, roll_t, pitch_t] + sub_coords
             f.write(','.join(f"{v:.3f}" for v in values) + '\n')