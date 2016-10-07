import urllib2
import os

import fredConf
filename='test.json'
#dir = os.path.dirname(filename)
jsonfile=open(filename,'w')

reqStart="http://api.stlouisfed.org/fred/"
reqNext="&api_key="
reqFile="&file_type="

fileType="json"

target="series/observations?series_id=GNPCA"


request=reqStart
request+=target
request+=reqNext
request+=key
request+=reqFile
request+=fileType

a=urllib2.urlopen(request).read()
print a
jsonfile.write(a)
