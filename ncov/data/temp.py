import datetime
import pandas as pd

df = pd.read_csv('allcity_2020{}.csv'.format(datetime.date.today().strftime('%m%d')))
beijing = df.loc[(df['provincename'] == '北京') & (df['cityname'] == '全市')]
beijing = beijing[['confirmed', 'suspected', 'dead', 'cured', 'time']]
beijing.to_csv('beijing_截至{}_24时.csv'.format((datetime.date.today() + datetime.timedelta(days=-1)).strftime('%m%d')), index=None)

shanghai = df.loc[(df['provincename'] == '上海') & (df['cityname'] == '全市')]
shanghai = shanghai[['confirmed', 'suspected', 'dead', 'cured', 'time']]
shanghai.to_csv('shanghai_截至{}_24时.csv'.format((datetime.date.today() + datetime.timedelta(days=-1)).strftime('%m%d')), index=None)

