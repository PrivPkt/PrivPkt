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
    

    
#at client side    
#if you don't want to send the whole h5 and only the update to the weight matrix use this 
modelUW = create_model(None) # we know the archtecture of the model 
modelJoint = create_model(None) # we know the archtecture of the model 

modelUW.load_weights('dp_model_3_validate_after_wasting_48hrs.h5') #load weights 
modelJoint.load_weights('McMasterU-1.h5') #load weights 


for layer in modelUW.layers:
    weightsUW = layer.get_weights() # list of numpy arrays
    
for layer in modelJoint.layers:
    weightsJoint = layer.get_weights() # list of numpy arrays
    
updateUW = weightsJoint - weightsUW








####################### Experiment below 




NUM_OF_CLIENTS=3
NUM_EPOCHS=NUM_OF_EPOS=5
batch_size = 1000
FREQ = 300

server_model=create_tehran_model()
client_models=[]
clients_weights_for_each_round=[]

#initialize

for epoch in range(NUM_OF_EPOS):
    clients_weights_for_each_round.append([])
    
for epoch in range(NUM_OF_EPOS):
    for host in range(NUM_OF_CLIENTS): 
        clients_weights_for_each_round[epoch].append([])

for host in range(NUM_OF_CLIENTS): 
    client_models.append(create_tehran_model())
    

val_gen = DataGen(
    "data/iscx_006d.mat", 
    idxfilter=lambda x: x % FREQ >= 295, 
    batch_size=batch_size
)


for epoch in range(NUM_OF_EPOS):
    print("EPOCH = " + str(epoch))
    for host in range(NUM_OF_CLIENTS):
         
        # fetch model
        np_model=client_models[host]
        
        # load joint weights into the model
        print("Loading joint weights into host", host)
        np_model.set_weights(server_model.get_weights())

        # create new generator sequence
        train_gen = DataGen(
            "data/iscx_006d.mat", 
            idxfilter=lambda x: x % FREQ == host * NUM_EPOCHS,
            batch_size=batch_size
        )

        # train again for 1 epoch
        print("Training on host", host)
        np_model.fit_generator(generator=train_gen, validation_data=val_gen, epochs=1, verbose=True)

        # save snapshot
        print("Saving weights for host", host)
        np_model.save_weights("fed_np_mesh_lite/fed_snapshot_host%02d_epoch%02d.h5" % (host, epoch))
        
        # save weights
        print("Save the weights in RAM")
        clients_weights_for_each_round[epoch][host]=np_model.get_weights()
    
        #loss, acc = np_model.evaluate_generator(val_gen)
        #print("Host " + str(host) + " Model ACC=" + str(acc))

    #server is now summing up the result of this epoch 
    weights=[]
    for host in range(NUM_OF_CLIENTS): 
        weights.append(clients_weights_for_each_round[epoch][host])
        
        
    joint=[] 
    for i in range(len(weights[0])): 
            joint.append(0)

    for host in range(NUM_OF_CLIENTS): 
        for i in range(len(weights[host])): 
                joint[i] += weights[host][i] #sum up 
        joint[i] = joint[i]/NUM_OF_CLIENTS
    server_model.set_weights(joint)
    
    loss, acc = server_model.evaluate_generator(val_gen)
    print("Server Model ACC="+ str(acc))
        
        
        
################# BASICALLY 


weights = [model.get_weights() for model in models]

new_weights = list()

for weights_list_tuple in zip(*weights):
    new_weights.append(
        [numpy.array(weights_).mean(axis=0)\
            for weights_ in zip(*weights_list_tuple)])

new_model.set_weights(new_weights)

