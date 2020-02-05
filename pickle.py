import pickle
file_name='/content/drive/My Drive/MEsc/Mia/fre/model_data'
fileObject = open(file_name,'wb')
pickle.dump(X_train,fileObject)
pickle.dump(y_train,fileObject)
pickle.dump(X_cv,fileObject)
pickle.dump(y_cv,fileObject)
pickle.dump(X_test,fileObject)
pickle.dump(y_test,fileObject)
fileObject.close()
