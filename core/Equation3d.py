import math

class Point :
    x = 0.0
    y = 0.0
    def __init__(self, x = 0.0,  y = 0.0) :
        self.x = x
        self.y = y
    def to_str(self):
        return "x:" + str(self.x) + ' y:' + str(self.y)
    def print(self):
        print(self.to_str())
class Point3d :
    x = 0.0
    y = 0.0
    z = 0.0
    def __add__(self,poi):
        return Point3d(self.x + poi.x,self.y + poi.y, self.z + poi.z)
    def __init__(self, x=0,  y=0,  z=0) :
        self.x = x
        self.y = y
        self.z = z
    def  xy(self) :
        return Point(self.x, self.y)
    def  xz(self) :
        return Point(self.x, self.z)
    def  yz(self) :
        return Point(self.y, self.z)
    def to_str(self) :
        return "x:" + str(self.x) + ' y:' + str(self.y) + ' z:' + str(self.z) 

class LineEquation2d :
    a = None
    b = 0.0
    c = 1
    # cy = ax + b
    # x = cy+b/a
    def set(self, x1,  y1,  x2,  y2) :
        if (x1 == x2) :
            # c = x1;
            self.a = -1
            self.b = x1
            self.c = 0
            return
        self.a = (y1 - y2) / (x1 - x2)
        self.b = (x2 * y1 - x1 * y2) / (x2 - x1)
    def __init__(self, p1,  p2) :
        if (p1.x == p2.x) :
            # c = x1;
            self.a = -1
            self.b = p1.x
            self.c = 0
            return
        self.a = (p1.y - p2.y) / (p1.x - p2.x)
        self.b = (p2.x * p1.y - p1.x * p2.y) / (p2.x - p1.x)
    def getX(self, y) :
        return (self.c * y - self.b) / self.a
    def getY(self, x) :
        return (self.a * x + self.b) / self.c
    def cos(self):
        return (1)/math.sqrt(((self.getY(1)-self.b)**2 + 1))
        
def  SolveEquation( line1,  line2) :
    result = [0.0] * (2)
    result[0] = (line1.b * line2.c - line1.c * line2.b) / (-line1.a * line2.c - line1.c * -line2.a)
    result[1] = (line1.b * -line2.a + line1.a * line2.b) / (line1.c * -line2.a + line1.a * line2.c)
    return result
def  SolveEquation_3d( line1,  line2,  res) :
    res_xy = SolveEquation(line1.line_xy, line2.line_xy)
    res_xz = SolveEquation(line1.line_xz, line2.line_xz)
    res.x = res_xz[0]
    res.y = res_xy[1]
    res.z = res_xz[1]
    return abs(res_xy[0] - res_xz[0])
class LineEquation3d :
    line_xy = None
    line_xz = None
    def __init__(self, point1:Point3d,  point2:Point3d):
        self.line_xy = LineEquation2d(Point(point1.x, point1.y), Point(point2.x, point2.y))
        self.line_xz = LineEquation2d(Point(point1.x, point1.z), Point(point2.x, point2.z))
        self.poi_buf = Point()
    '''
    def __init__(self) :
        self.line_xy = LineEquation2d()
        self.line_xz = LineEquation2d()
        self.poi_buf = Point()
    '''
    def set(self, point1,  point2) :
        self.line_xy.set(point1.x, point1.y, point2.x, point2.y)
        self.line_xz.set(point1.x, point1.z, point2.x, point2.z)
    def getPoint(self, val:dict) :
        result = [None,None,None]
        if 'x' in val:
            result[0] = val['x']
            result[1] = self.line_xy.getY(result[0])
            result[2] = self.line_xz.getY(result[0])
        elif 'y' in val:
            result[1] = val['y']
            result[0] = self.line_xy.getX(result[1])
            result[2] = self.line_xz.getY(result[0])
        elif 'z' in val:
            result[2] = val['z']
            result[0] = self.line_xz.getX(result[2])
            result[1] = self.line_xy.getY(result[0])
        return result
    def getPointByZ(self, z,  result) :
        result.x = self.line_xz.getX(z)
        result.y = self.line_xy.getY(result.x)
        result.z = z


def  Distance( poi1,  poi2) :
    return math.sqrt(math.pow(poi1.x - poi2.x,2) + math.pow(poi1.y - poi2.y,2))

def  getQuadrant( target,  positive) :
    ver = target.y > positive.y
    hor = target.x > positive.x
    if (ver) :
        if (hor) :
            return 4
        else :
            return 3
    elif(hor) :
        return 1
    else :
        return 2
def  MaxOfArray( data, length, id = None) :
    if id == None:
        id = [i for i in range(length)]
    max = 0
    i = 0
    while (i < length) :
        if (data[max] < data[i]) :
            max = i
        i += 1
    return id[max]


class CurveEquation :
    # f(x) = ax^2 + bx + c;
    a = 0.0
    b = 0.0
    c = 0.0
    def remake(self, p1,  p2,  p3) :
        temp1 = (p2.y - p1.y) / (p2.x - p1.x)
        temp2 = (p3.y - p1.y) / (p3.x - p1.x)
        self.a = (temp1 - temp2) / (p2.x - p3.x)
        self.b = temp1 - self.a * (p1.x + p2.x)
        self.c = p1.x - self.b * p1.x - self.a * p1.x * p1.x
    def __init__(self, p1,  p2,  p3) :
        temp1 = (p2.y - p1.y) / (p2.x - p1.x)
        temp2 = (p3.y - p1.y) / (p3.x - p1.x)
        self.a = (temp1 - temp2) / (p2.x - p3.x)
        self.b = temp1 - self.a * (p1.x + p2.x)
        self.c = p1.y - self.b * p1.x - self.a * p1.x * p1.x
    def toStandard(self) :
        res = CurveEquationStandard()
        return res
    def f(self, x) :
        return self.a * math.pow(x,2) + self.b * x + self.c

class CurveEquationStandard:
    # f(x) = a(x-h)^2 + k;
    a = 0.0
    h = 0.0
    k = 0.0
    def __init__(self, p1,  p2,  p3) :
        equation = CurveEquation(p1, p2, p3)
        self.a = equation.a
        self.h = -equation.b / (2 * equation.a)
        self.k = -(equation.b * equation.b - 4 * equation.a * equation.c) / (4 * equation.a)

    def f(self,x):
        return self.a * (x - self.h) * (x - self.h) + self.k

def  ApproxEqu( obj,  tar,  deviation) :
    return abs(obj - tar) < deviation
def  Value2percentage( max,  min,  x) :
    return (x - min) / float((max - min))
def LinearScaler(object, object_min, object_max, target_min, target_max):
    return (object-object_min)/(object_max-object_min) * (target_max-target_min) + target_min

if __name__ == "__main__":
    c1 = CurveEquation(Point(0,0), Point(1,1), Point(2,5))
    c2 = c1.toStandard()
    c3 = CurveEquationStandard(Point(0,0), Point(1,1), Point(2,5))
    print(c1.f(2), c2.f(2), c3.f(2))
    print(c1.a, c1.b, c1.c)
    print(c2.a, c2.h, c2.k)
    print(c3.a, c3.h, c3.k)

    