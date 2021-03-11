# ner

https://drive.google.com/file/d/16fpyb1O0_0-ZBc_EHYN5OscUUB95NHNX/view?usp=sharing


pred_train = []
for i in range(0,1000):
  a = model.predict({'input_ids': x['input_ids'][i],'token_type_ids':x['token_type_ids'][i],'attention_mask':x['attention_mask'][i]})
  a = np.transpose(a,(1,0,2))
  a = a[0]
  pred_train.append(list(map(lambda x : list(x).index(max(list(x))), a )))

print(classification_report( y[0:1000].ravel()   ,  pred_train.ravel()    )) 
