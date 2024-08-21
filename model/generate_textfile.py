import os

count=0
file=open("test_address.txt","w")
for path,subdir,files in os.walk('D:\\Major-Project\\CNNnLSTM\\Project\\UCF-101_test_features'):
	for filename in files:
		f=os.path.join(path,filename)
		file.write(str(f)+'\n')

print("done")	
