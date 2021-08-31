import datetime as dt
now = dt.datetime.now().time()
nows = []
x = 0
delta = dt.timedelta(seconds=x)
for i in range(31):
    nows.append((dt.datetime.combine(dt.date(1,1,1),now) + delta).time())
    x +=0.1
    delta = dt.timedelta(seconds=x)
print(nows)