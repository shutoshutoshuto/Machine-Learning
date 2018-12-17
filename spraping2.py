import os
from urllib import request as req
from urllib import error
from urllib import parse
import bs4


keyward = "ねこ"
if not os.path.exists(keyward):
    os.mkdir(keyward)
    
urlkeyward = parse.quote(keyward)