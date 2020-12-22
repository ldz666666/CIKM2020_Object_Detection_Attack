import json
import cv2

name='output_9_4'

dict_rcnn=json.load(open('./output_data/rcnn_{}_all.json'.format(name),'rb'))
dict_yolo=json.load(open('./output_data/yolo_{}_all.json'.format(name),'rb'))
dict_connected=json.load(open('./output_data/connected_domin_score_{}.json'.format(name),'rb'))
#print(dict_rcnn)
zero_rcnn={pair[0]:pair[1] for pair in sorted(dict_rcnn.items()) if pair[1]<0.1 }
zero_yolo={pair[0]:pair[1] for pair in sorted(dict_yolo.items()) if pair[1]<0.1 }
zero_both={pair[0]:pair[1] for pair in sorted(zero_rcnn.items()) if pair in zero_yolo.items() }

zero_domain={pair[0]:pair[1] for pair in sorted(dict_connected.items()) if pair[1]<0.1 }

dict_total={pair[0]:(pair[1]+dict_yolo[pair[0]]) for pair in sorted(dict_rcnn.items()) }

print('zero rcnn is')
print(zero_rcnn)
print('len zero rcnn',len(zero_rcnn))

print('zero yolo is')
print(zero_yolo)
print('len zero yolo',len(zero_yolo))

print('zero both is')
print(zero_both)
print('len zero both',len(zero_both))

print('sum rcnn is')
print(sum(dict_rcnn.values()))
print('sum yolo is')
print(sum(dict_yolo.values()))

print('zero domain is')
print(len(zero_domain.values()))

json.dump(dict_total,open('sum_{}.json'.format(name),'w'))