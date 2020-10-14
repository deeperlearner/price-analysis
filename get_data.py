import os
from datetime import datetime

import requests

from parser import args

data_path = "./dataset"
if not os.path.exists(data_path):
    os.mkdir(data_path)

CRYPTO = args.crypto
TICK   = args.tick
DATE = datetime.today().strftime("%Y%m%d")

filename = "{}_{}_{}.csv".format(CRYPTO, TICK, DATE)
file_path = os.path.join(data_path, filename)

def Download_Data():
    if os.path.isfile(file_path):
        print(file_path + " was already Downloaded!")
    else:
        # https://www.cryptodatadownload.com/data/northamerican/
        url = "https://www.cryptodatadownload.com/cdd/Coinbase_{}USD_{}.csv".format(CRYPTO, TICK)
        print("Retrieving Data from %s..." % url)
        raw_text = requests.get(url, verify='cryptodatadownload.cer').text
        head, raw_text = raw_text.split('\n', 1)

        with open(file_path, 'w') as text_file:
            text_file.write(raw_text)

if __name__ == '__main__':
    Download_Data()