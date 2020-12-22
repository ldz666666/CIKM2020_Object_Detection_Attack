import json
import cv2
import shutil
import os

file_dir='./select1000_new/'
files=os.listdir(file_dir)
files.sort()

name='9_3_choose_2'

save_dir='./output_{}/'.format(name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

dict_1=json.load(open('sum_all_9_4.json','rb'))
dict_2=json.load(open('sum_4000_pix.json','rb'))
dict_3=json.load(open('sum_3000_pix.json','rb'))
dict_4=json.load(open('sum_2000_pix.json','rb'))
dict_5=json.load(open('sum_1000_pix.json','rb'))

img_list=['/eval_code/output_9_3_choose_1/','/eval_code/output_4000_pix/','/eval_code/output_3000_pix/','/eval_code/output_2000_pix','/eval_code/output_1000_pix']
dict_list=[dict_1,dict_2,dict_3,dict_4,dict_5]

maxn=0
for d in dict_list:
    maxn=max(sum(d.values()),maxn)
print('before , max sum score is')
print(maxn)


dict_sta={}

dict_total={}

for filename in files:
    print('dealing',filename)
    ind=0
    score=dict_list[ind][filename]
    for i in range(len(dict_list)):
        if filename in dict_list[i]:
            if dict_list[i][filename]>score:
                ind=i
    shutil.copy(os.path.join(img_list[ind],filename),os.path.join(save_dir,filename))
    dict_total[filename]=dict_list[ind][filename]
    if ind not in dict_sta.keys():
        dict_sta[ind]=1
    else:
        dict_sta[ind]+=1

max_later=sum(dict_total.values())
#print(dict_total)
print(max_later)
print(max_later-maxn)
print('save finished')
json.dump(dict_total,open('sum_all_{}.json'.format(name),'w'))

for k,v in sorted(dict_sta.items()):
    print(img_list[k],v)



