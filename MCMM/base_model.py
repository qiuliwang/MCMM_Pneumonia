import os
import numpy as np
import tensorflow as tf
import pickle
import copy
from tqdm import tqdm
import scipy.misc

from utils.nn import NN

from utils.misc import ImageLoader, CaptionData, TopN
from sklearn.metrics import roc_auc_score,recall_score
class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        self.image_shape = [512, 512, 3]
        self.nn = NN(config)
        self.global_step = tf.Variable(0,
                                       name = 'global_step',
                                       trainable = False)
        self.build()
        self.record = open('lossrecord_guiyihua_oneloss.txt', 'w')
        self.predrecord = open('predrecord2_guiyihua_oneloss'+ self.config.cnn + '_'+ str(self.config.num_lstm_units) + '_'+'_.csv', 'w')

    def build(self):
        raise NotImplementedError()

    def train(self, sess, train_data, test_data, dataobj, transign):
        """ Train the model. """
        print("Training the model...")
        config = self.config

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)

        record = self.record
        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):

            for onepatient in tqdm(train_data, desc = 'data'):

                slices, sex, age, wordsvec = dataobj.getOnePatient(onepatient, transign)
                if 'nor' in onepatient[1]:
                    onelabel = [[0, 1]]
                else:
                    onelabel = [[1, 0]]
                sex = [[sex]]
                age = [[age]]
                feed_dict = {self.images: slices, self.real_label: onelabel, self.sex: sex, self.age: age, self.complaint: wordsvec}            

                _, summary, global_step = sess.run([self.opt_op,
                                                    self.summary,
                                                    self.global_step],
                                                    feed_dict=feed_dict)

            print('@@@@@@@@@@@@@@@@@@@@@@@@@')
            temploss = 0.0
            correctpercent = 0.0
            for testpatient in test_data:
                if 'nor' in testpatient[1]:
                    onelabel = [[0, 1]]
                else:
                    onelabel = [[1, 0]]  
                slices, sex, age, wordsvec = dataobj.getOnePatient(testpatient, False)
                sex = [[sex]]
                age = [[age]]
                feed_dict = {self.images: slices, self.real_label: onelabel, self.sex: sex, self.age: age, self.complaint: wordsvec}            
                loss, correctpred = sess.run([self.loss, self.correct_pred], feed_dict=feed_dict)
                temploss += loss
                if correctpred == True:
                    correctpercent += 1
                else:
                    z = 0
            print('loss: ', temploss / len(test_data))
            print('acc: ', correctpercent / len(test_data))
            record.write('loss: ' + str(temploss / len(test_data)) + ', ')
            record.write('acc: ' + str(correctpercent / len(test_data)) + '\n')

            train_writer.add_summary(summary, global_step)


        self.save()
        # record.close()
        train_writer.close()
        print("Training complete.")

    def getLabel2(self, onepatient, alllabel):
        for onelabel in alllabel:
            if onepatient == onelabel[0]:
                if int(onelabel[1]) == 1:
                    return([0, 1])
                elif int(onelabel[1]) == 0:
                    return([1, 0])

                break
            
    def getLabel(self, onepatient, alllabel):
        label = []
        for onelabel in alllabel:
            if onepatient == onelabel[0]:
                if int(onelabel[1]) == 1:
                    label.append([0, 1])
                elif int(onelabel[1]) == 0:
                    label.append([1, 0])

                break
            
        return label

    def test(self, sess, testdata, dataobj, sign):
        """ Test the model using any given images. """
        print("Testing the model ...")
        config = self.config
        predrecord = self.predrecord
        predrecord.write('id,ret\n')
        correctpercent = 0
        TP=0
        TN=0
        FP=0
        FN=0
        label=[]
        pre=[]
        for onetestpatient in tqdm(testdata):
            if 'nor' in onetestpatient[1]:
                onelabel = [[0, 1]]
            else:
                onelabel = [[1, 0]]
            label.append(np.argmin(onelabel))

            slices, sex, age, wordsvec = dataobj.getOnePatient(onetestpatient)
            sex = [[sex]]
            age = [[age]]           
            feed_dict = {self.images: slices, self.real_label: onelabel, self.sex: sex, self.age: age, self.complaint: wordsvec}

            prediction, correct_pred = sess.run([self.prediction, self.correct_pred], feed_dict=feed_dict)
            pre.append(1-prediction)
            if correct_pred == True: 
                correctpercent += 1
                if 'nor' in onetestpatient[1]:
                    TN+=1
                else:
                    TP+=1
            else:
                if 'nor'  in onetestpatient[1]:
                    FP+=1
                else:
                    FN+=1
                predrecord.write(onetestpatient[0]+','+onetestpatient[1]+'\n')
        print(correctpercent)
        print(len(testdata))
        print('Sensitivity',float(TP)/float(TP+FN))
        print('Specificity',float(TN)/float(FP+TN))
        print('AUC',roc_auc_score(label,pre))
        
        print('accuracy: ', float(correctpercent) / len(testdata)) 
        # predrecord.close()

        print("Test completed.")

    def save(self):
        """ Save the model. """
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

        print((" Saving the model to %s..." % (save_path +".npy")))
        np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir,
                                     str(global_step)+".npy")

        print("Loading the model from %s..." %save_path)
        data_dict = np.load(save_path).item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." %count)
    

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." %data_path)
        data_dict = np.load(data_path).item()
        count = 0
        for op_name in tqdm(data_dict):
            with tf.variable_scope(op_name, reuse = True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                    except ValueError:
                        pass
        print("%d tensors loaded." %count)
