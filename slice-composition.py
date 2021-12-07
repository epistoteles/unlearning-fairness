from collections import Counter
from data.UTKFaceDataset import UTKFaceDataset


d = UTKFaceDataset(split='all')
c = Counter([(x.race, x.changed_privacy_settings) for x in d.faces])

did = sorted([x for x in c.items() if x[0][1]], key=lambda x: x[0][0])
didnt = sorted([x for x in c.items() if not x[0][1]], key=lambda x: x[0][0])

sum_did = sum([x[1] for x in did])
sum_didnt = sum([x[1] for x in didnt])

props_did = [x[1]/sum_did for x in did]
props_didnt = [x[1]/sum_didnt for x in didnt]

print(['white', 'black', 'asian', 'indian', 'other'])
print(props_did)
print(props_didnt)

"""
[0.44615648005478514, 0.17086115391200138, 0.14021571648690292, 0.17788049991439822, 0.06488614963191235]
[0.41820624790057104, 0.1975142761168962, 0.14640017915127085, 0.1643712910088456, 0.0735080058224163]
"""