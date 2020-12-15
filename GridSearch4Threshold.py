#Get output of AE for Validation Set and Attack Set
ae1_cv=ae2.predict(cv_x.reshape([-1,784]))
ae1_adv=ae2.predict(attacked_data.reshape([-1,784]))

#Calculate the 2nd norm reconstruction error
cv_res = np.linalg.norm(ae1_cv-cv_x.reshape([-1,784]),axis=1)
adv_res = np.linalg.norm(ae1_adv-attacked_data.reshape([-1,784]),axis=1)

#Check mean values of the reconstruction error
print(np.mean(adv_res))
print(np.mean(cv_res))
#Set seaching lower bound a, and upper bound b
a=3000
b=10000


from sklearn.metrics import f1_score
#The true label y_hat: validation set is innocent, adv set is adversarial
y_hat = np.ones([cv_res.shape[0]+adv_res.shape[0]])
y_hat[0:cv_res.shape[0]] = 0

#Variables to save the best result
best_th = 0
best_score = 0
best_h = None

#for loop for grid search
for t in range(a,b,1):
  th = t*0.001
  #Making predictions based on the threshold
  cv_pred_y = cv_res>th
  adv_pred_y = adv_res>th
  h_hat = np.ones([cv_res.shape[0]+adv_res.shape[0]])
  h_hat[0:cv_res.shape[0]] = cv_pred_y
  h_hat[cv_res.shape[0]:] = adv_pred_y
  
  #Using macro F1 score to evaluate the performance
  z = f1_score(y_hat, h_hat, average='macro')
  print(z)
  
  #We hope the accuracy on adversarial case is greater than 0.9
  acc_adv = np.sum(adv_pred_y)/adv_pred_y.shape
  #print(acc_adv)
  if z>best_score and acc_adv>0.9:
    best_score = z
    best_th = th
    best_h = h_hat
