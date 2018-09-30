import pandas as pd

files_wanted = pd.read_csv('phx_lidar.csv').values.tolist()
files_wanted = [item for sublist in files_wanted for item in sublist]

files = os.listdir('las')

for f in files:
    if not any(f[4:8] in s for s in files_wanted):
        # files.remove(f)
        os.remove('las/' + f)



###
# download por
import urllib.request
import pandas as pd
# Download the file from `url` and save it locally under `file_name`:
files_wanted = pd.read_csv('por_las_url.csv').values.tolist()
files_wanted = [item for sublist in files_wanted for item in sublist]
for f in files_wanted:
    url = f[0:-2]
    file_name = f[0:-2].split('/')[-1]
    urllib.request.urlretrieve(url, file_name)
