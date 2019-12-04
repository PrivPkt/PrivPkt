import numpy as np
from absl import app
from absl import flags
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data
from mia.serialization import BaseModelSerializer

class MySerializer(BaseModelSerializer):
    
    def __init__(self, model_fn, *args, **kwargs):
        super().__init__(model_fn,  *args, **kwargs)
    
    def load(self, model_id):
        model = self.model_fn()
        model.load_weights("{}/smb_{}.h5".format(self.models_path, model_id))
        return model
    
    def save(self, model_id, model):
        model.save_weights("{}/smb_{}.h5".format(self.models_path, model_id))
        
SHADOW_DATASET_SIZE = 19000
ATTACK_TEST_DATASET_SIZE = 19000
num_shadows=2
attack_epochs=target_epochs=6


batch_size = 1000
FREQ = 300

val_gen = DataGen(
    "data/iscx_006d.mat", 
    idxfilter=lambda x: x % 55 ==2, 
    batch_size=batch_size
)

data=[]
for i in range(len(val_gen)):
    data.append(val_gen[i])
X_test = np.concatenate(tuple([d[0] for d in data]))
y_test = np.concatenate(tuple([d[1] for d in data]))


smb_serial = MySerializer(
    model_fn=create_dpsgd_model_f, 
    prefix="./smb_DPSGD__model_weights_timeS_"+"ZENA"
)


# Train the shadow models.
smb = ShadowModelBundle(
    create_dpsgd_model_f,
    shadow_dataset_size=SHADOW_DATASET_SIZE, #dataset size for shadow
    num_models=num_shadows, #how many shadows 
    serializer=smb_serial
)

# We assume that attacker's data were not seen in target's training.
attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
    X_test, y_test, test_size=0.2)
print(attacker_X_train.shape, attacker_X_test.shape)

print("Training the shadow models...")

saved=0 # MANUALLY


if saved ==0: 
    #if model is not saved 
    X_shadow, y_shadow = smb.fit_transform(
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=target_epochs,
            verbose=True,
            validation_data=(attacker_X_test, attacker_y_test),
            batch_size = 100
        ),
    )



else:
    # if model is saved 
    X_shadow, y_shadow = smb._transform(
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=target_epochs,
            verbose=True,
            validation_data=(attacker_X_test, attacker_y_test),
            batch_size = 100),)







amb_serializer=MySerializer(model_fn=attack_model_fn, prefix="./amb_DPSGD__model_weights_zena")
# ShadowModelBundle returns data in the format suitable for the AttackModelBundle.

amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES, serializer=amb_serializer)
# Fit the attack models.
print("Training the attack models...")
amb.fit(X_shadow, y_shadow, fit_kwargs=dict(epochs=attack_epochs, verbose=False))













import json
#host=0
attack_accuracy_class_perhost=[]
for host in range(10):
    print("host == " + str(host))
    train_gen = DataGen(
        "data/iscx_006d.mat", 
        idxfilter=lambda x: x % FREQ == host * 10,
        batch_size=batch_size
    )
    data=[]
    for i in range(len(train_gen)):
        data.append(train_gen[i])
    X_in3 = np.concatenate(tuple([d[0] for d in data]))
    y_in3 = np.concatenate(tuple([d[1] for d in data]))
    target_model=models[host]

    X_victim = X_in3
    y_victim = y_in3
    X_not_victim=X_test
    y_not_victim=y_test
    y_victim_nc = np.argmax(y_victim, axis=1)
    y_not_victim_nc = np.argmax(y_not_victim, axis=1)

    class_indexes = [[] for _ in range(8)]
    class_indexes_notv = [[] for _ in range(8)]

    for i, c in enumerate(y_victim_nc):
        class_indexes[c].append(i)

    for i, c in enumerate(y_not_victim_nc):
        class_indexes_notv[c].append(i)

    attack_accuracy_class={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}

    for i in range(NUM_CLASSES):
        X_victim = X_in3
        y_victim = y_in3

        X_not_victim=X_test
        y_not_victim=y_test


        X_class = X_victim[class_indexes[i], :]
        y_class = y_victim[class_indexes[i], :]

        X_class_notv = X_not_victim[class_indexes_notv[i], :]
        y_class_notv = y_not_victim[class_indexes_notv[i], :]

        # Prepare examples that were in the training, and out of the training.
        data_in = X_class[:ATTACK_TEST_DATASET_SIZE], y_class[:ATTACK_TEST_DATASET_SIZE]
        data_out = X_class_notv[:ATTACK_TEST_DATASET_SIZE], y_class_notv[:ATTACK_TEST_DATASET_SIZE]

        # Compile them into the expected format for the AttackModelBundle.
        attack_test_data, real_membership_labels = prepare_attack_data(
            target_model, data_in, data_out
        )

        # Compute the attack accuracy.
        attack_guesses = amb.predict(attack_test_data)
        attack_accuracy = np.mean(attack_guesses == real_membership_labels) # sum / len 

        result=give_me_results(attack_guesses,real_membership_labels)

        print(str(i) + "    " + str(result)) 
        attack_accuracy_class[i].append(result)

    attack_accuracy_class_perhost.append(attack_accuracy_class)
    with open('DPSGD_saved_attack_accuracy_class_perhost' + str(host)+'.txt', 'w') as file:
        file.write(json.dumps(attack_accuracy_class))







def give_me_results(attack_guesses,real_membership_labels):
    pred_labels=attack_guesses
    true_labels=real_membership_labels

    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

    print ('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))

    print("acc = " + str((TP+TN)/(TP+TN+FP+FN)))
    print("precision = " + str((TP)/(TP+FP)))
    print("recall = " + str((TP)/(TP+FN)))
    
    acc= (TP+TN)/(TP+TN+FP+FN)
    prec=(TP)/(TP+FP)
    rec=(TP)/(TP+FN)
    return [acc,prec,rec]
