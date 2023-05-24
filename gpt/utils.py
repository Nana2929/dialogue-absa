from typing import List, Dict
import json

def write_jsonstr(data: dict, fp: str):
    with open(fp, "w") as f:
        dictstr = json.dumps(data, indent=4)
        f.write(dictstr)

def write_json(datas: List[Dict], fp: str):
    with open(fp, 'w') as f:
        json.dump(datas, f, indent=4, ensure_ascii=False) 
