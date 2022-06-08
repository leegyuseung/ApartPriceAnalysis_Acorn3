'''
Created on 2022. 6. 8.

@author: u
'''
from map.models import Addrdata, Addrapt
import pandas as pd
def dbconnect(addr):
        
        
    datas = Addrdata.objects.filter(addr__contains=addr).values()    
    df = pd.DataFrame(datas)
    df['price']= (df['price']/df['area']) * 3.3
    df2 = df.loc[:,['price','ymd','gu']]
    print(df2)

dbconnect('')