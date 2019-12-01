# Read h5 weights 
modelUW = create_model() # we know the archtecture of the model 
modelMAC = create_model() # we know the archtecture of the model 

modelUW.load_weights('UWaterloo-1.h5') #load weights 
modelMAC.load_weights('McMasterU-1.h5') #load weights 

models=[modelUW,modelMAC]
models_weights={}
for m in models: 
    models_weights[m]=m.get_weights()

joint=[0,0,0,0,0,0,0,0,0,0] #10
for m in models: 
    for i in range(len(models_weights[m])): 
            joint[i] += models_weights[m][i] #sum up 
    joint[i] = joint[i]/10 #avg the weights 

modelJ = create_model(joint) #we know the architecture #note this time we pass joint weights where the function will do model.set_weights(weights)

#we can use the joint model now modelJ 




loss, acc = model3.evaluate_generator(val_gen) 
print(acc)

our_acc_threshold = 0.91

if acc > our_acc_threshold: 
    #update weights in the models 
    for m in models: 
        m.set_weights(joint)
    #save h5 weights 
    modelJ.save_weights('Join-1.h5')
    
