# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:42:42 2017

@author: jmonteiro
"""
import urllib.request
import sys
import datetime

#startdate = sys.argv[1]
#enddate = sys.argv[2]
#download_dir = sys.argv[3]
startdate = '20170501'
enddate = '20170531'
download_dir = 'C:/tmp/deep-mind'

dateFormat = '%Y%m%d'

def download(date):
	url = 'ftp://ftp.bmf.com.br/MarketData/Bovespa-Vista/NEG_' + date + '.zip'
	filepath = download_dir + '/NEG_' + date + '.zip'
	try:
		urllib.request.urlretrieve(url, filepath)
		print("Downloaded '" + url + "' to '" + filepath+ "'.")
	except:
		print("It was not possible to download '" + url + "'.")

start = datetime.datetime.strptime(startdate, dateFormat)
end = datetime.datetime.strptime(enddate, dateFormat)
current = start
while(current <= end):
	download(current.strftime(dateFormat))
	current += datetime.timedelta(days=1)