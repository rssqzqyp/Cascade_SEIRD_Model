# import time
import datetime as dt
from ops import read_data
from train import train_with_city_data
citys = ['湖北', '武汉', '深圳', '全国']
N_inits = [59170000., 2870000., 13026600.]
td = dt.datetime.today()
datetime = '02-28'
time = '0228'
yesterday = '02-27'
paths = [
    './ncov/data/hubei_截至' + time + '_24时.csv',
    './ncov/data/wuhan_截至' + time + '_24时.csv',
    './ncov/data/shenzhen_截至' + time + '_24时.csv',
    './ncov/data/nation_截至' + time + '_24时.csv'
]
param_paths_yes = [
    'models/' + 'hubei/' + yesterday, 'models/' + 'wuhan/' + yesterday,
    'models/' + 'shenzhen/' + yesterday, 'models/' + 'china/' + yesterday
]
print("today:", datetime)
print("time:", time)
print("yesterday:", yesterday)

i = 0
data = read_data(paths[i])
city_name = citys[i]
param_path = param_paths_yes[i]
# param_path=''
N = 59170000.
model_city_date_path = train_with_city_data(data,
                                            N,
                                            datetime,
                                            city_name,
                                            max_epoches=10000,
                                            is_train=True,
                                            load_param_save=True,
                                            lr_init=0.00001,
                                            param_path=param_path)
