from collections import Counter
from data.UTKFaceDataset import UTKFaceDataset
import random

d = UTKFaceDataset(split='all')

# print(len(d.faces))
# to_be_removed = []
# for race in range(5):
#     for age_bin in range(7):
#         candidates = [f for f in d.faces if f.race == race and f.age_bin == age_bin]
#         to_be_removed += random.sample(candidates, 9)
# filtered_faces = [f for f in d.faces if f not in to_be_removed]
# print(len(filtered_faces))

c = Counter([(x.race, x.changed_privacy_settings) for x in d.faces])  # d.faces or filtered_faces

did = sorted([x for x in c.items() if x[0][1]], key=lambda x: x[0][0])
didnt = sorted([x for x in c.items() if not x[0][1]], key=lambda x: x[0][0])
all = sorted(Counter([x.race for x in d.faces]).items(), key=lambda x: x[0])

sum_did = sum([x[1] for x in did])
sum_didnt = sum([x[1] for x in didnt])
sum_all = sum([x[1] for x in all])

props_did = [x[1]/sum_did for x in did]
props_didnt = [x[1]/sum_didnt for x in didnt]
props_all = [x[1]/sum_all for x in all]

print(['white', 'black', 'asian', 'indian', 'other'])
print(props_did)
print(props_didnt)
print(props_all)

"""
props_did = [0.44615648005478514, 0.17086115391200138, 0.14021571648690292, 0.17788049991439822, 0.06488614963191235]
props_didnt = [0.41820624790057104, 0.1975142761168962, 0.14640017915127085, 0.1643712910088456, 0.0735080058224163]

props_did = [0.42901500776263585, 0.16888045540796964, 0.15094014145247542, 0.18130067276177333, 0.06986372261514577]
props_didnt = [0.4238244163967385, 0.19809002568971296, 0.14291299005919803, 0.16329721880933765, 0.07187534904501285]

props_did = [0.43127298196462965, 0.17387497811241465, 0.14936088250744178, 0.18333041498861846, 0.06216074242689547]
props_didnt = [0.4271086722860214, 0.19630027719635684, 0.14244498500876845, 0.16207501272840413, 0.07207105278044917]
"""