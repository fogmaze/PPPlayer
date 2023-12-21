import math

def c(cmd) :
    arm_length = 1.2
    arm_height = 1

    if abs((float(cmd[2])-arm_height)/arm_length) > 1 :
        print("out of range")
        return None
    rad1 = math.asin((float(cmd[2])-arm_height)/arm_length)
    rad2 = math.pi - rad1

    arm_borad1 = math.cos(rad1) * arm_length
    arm_borad2 = -arm_borad1

    base_pos1 = round(float(cmd[1]-arm_borad1)/0.0008)
    base_pos2 = round(float(cmd[1]-arm_borad2)/0.0008)

    now_baes_pos = 0
    now_rad = 0

    if abs(rad1 - now_rad) < abs(rad2 - now_rad) and -756 < base_pos1 < 756 and -0.25*math.pi < rad1 < 1.25*math.pi:
        rad = rad1
        base_pos = base_pos1
    elif -756 < base_pos2 < 756 and -0.25*math.pi < rad2 < 1.25*math.pi :
        rad = rad2
        base_pos = base_pos2
    else :
        print(rad1, rad2, base_pos1, base_pos2)
        return None

    deg = 225 - rad * 180 / math.pi
    return deg, base_pos


print(c([0.0, -0.5, .5]))