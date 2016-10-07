import pandas as pd
data = pd.io.stata.read_stata('~/Projects/predictions/bdm2s2_leader_year_data.dta')
data.to_csv('my_stata_file.cvs')
