import pandas as pd

unmatch_links_df = pd.read_csv('unmatch_links.csv')

print(unmatch_links_df.shape)

temp_df = unmatch_links_df.iloc[:60, :]

print(temp_df.shape)
