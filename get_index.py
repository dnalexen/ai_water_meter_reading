import pandas as pd
import math
import numpy as np


def get_index(temp_df):

  distance = []

  max_iter = temp_df.shape[0]

  index_liters = temp_df[temp_df['cls_max_id']==11]. index
  index_counter = temp_df[temp_df['cls_max_id']==10]. index

  x1_liters = temp_df.iloc[index_liters]['x1']
  y1_liters = temp_df.iloc[index_liters]['y1']

  x1_counter = temp_df.iloc[index_liters]['x1']
  y1_counter = temp_df.iloc[index_liters]['y1']


  if len(index_liters) == 1 :
    for i in range(max_iter):
      dist_x1y1 = math.sqrt(math.pow((temp_df.iloc[i]['x1'] - temp_df.iloc[index_liters]['x1']), 2) + \
                            math.pow((temp_df.iloc[i]['y1'] - temp_df.iloc[index_liters]['y1']), 2))

      distance.append(dist_x1y1)

    indexes_ordered = np.argsort(distance)[::-1]

  elif len(index_counter) == 1:
    for i in range(max_iter):
      dist_x1y1 = math.sqrt(math.pow((temp_df.iloc[i]['x1'] - temp_df.iloc[index_counter]['x1']), 2) + \
                            math.pow((temp_df.iloc[i]['y1'] - temp_df.iloc[index_counter]['y1']), 2))

      distance.append(dist_x1y1)

      indexes_ordered = np.argsort(distance)

  else:
      indexes_ordered = temp_df.index


  class_ordered = [temp_df.iloc[index]['cls_max_id'] for index in indexes_ordered]


  class_ordered = [str(digit) for digit in class_ordered if digit not in [10, 11]]

  return int(''.join(class_ordered))
