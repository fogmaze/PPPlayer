import csv

_ang_data_rev = {}
_ang_data = {}
key = []
data_rows = 0
with open("servo_angle.csv") as f:
    reader = csv.reader(f)
    first = next(reader)
    for i in first :
        _ang_data_rev[i] = 0
        key.append(i)
    for row in reader :
        data_rows += 1
        for i, d in enumerate(row):
            _ang_data_rev[key[i]] += float(d)
        
    for k in key :
        _ang_data_rev[k] /= data_rows
    
    for k in key :
        _ang_data[_ang_data_rev[k]] = float(k)

print(_ang_data)
pass