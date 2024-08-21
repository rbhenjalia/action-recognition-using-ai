import os

count=0
file=open("address1.csv","w")
for path,subdir,files in os.walk('/home/abhishek/Major Project/UCF-101_train_features'):
	for filename in files:
		f=os.path.join(path,filename)
		file.write(str(f)+os.linesep)

print("done")			
	