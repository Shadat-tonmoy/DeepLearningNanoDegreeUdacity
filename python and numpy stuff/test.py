from sklearn import tree 

data = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],
		[159,55,37],[171,75,42],[181,85,43]]

lable = ['male','female','female','female','male','male','male','female','male','female','male']

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(data,lable)

prediction = classifier.predict([[190,70,43]])

print ("Predited Gender is " + prediction[0])