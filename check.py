import json
for name in ("p00","p01"):
    d=json.load(open("out/"+name+".out"))
    rows=d["trophies"]
    ok=all(len(r["q"])==5 and all(0<=v<=100 for v in r["q"]) and all(r["q"][i]<=r["q"][i+1] for i in range(4)) for r in rows)
    print(name,"rows",len(rows),"valid",ok,"sample",rows[0],rows[53])
