#!/usr/bin/env python
# coding: utf-8

# In[11]:


from tkinter import *
from flask import Flask
app = Flask(__name__)

import numpy as np
import pandas as pd
p = 0
q= 0 
r = 0
l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']


disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]

for i in range(0,len(l1)):
    l2.append(0)


# In[12]:


df=pd.read_csv(r"traindata.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)



X= df[l1]
y = df[["prognosis"]]
np.ravel(y)



tr=pd.read_csv(r"testdata.csv")

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr["prognosis"]
print(list(y_test.values))


# In[13]:


cities = ['Bangalore','Chennai','Hyderabad','Vizag','Tirupati']

bnglr = pd.read_csv(r"bangalore.csv")
Bangalore = bnglr['Hospital']

chn = pd.read_csv(r"Chennai.csv")
Chennai = chn['Hospital']

hyd = pd.read_csv(r"HYD.csv")
Hyderabad = hyd['Hospital']

viz = pd.read_csv(r"Vizag.csv")
Vizag = viz['Hospital']

tpt = pd.read_csv(r"Tirupati.csv")
Tirupati = tpt['Hospital']


# In[14]:


def DecisionTree():

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier() 
    clf3 = clf3.fit(X.values,y)

    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test,y_pred))

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1
    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t1.delete("1.0", END)
        t1.insert(END,disease[a])
        p = a
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")


def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X.values,np.ravel(y))

    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test,y_pred))
    
    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1
    
    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
        q = a
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")


def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X.values,np.ravel(y))

    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test,y_pred))

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1
   
    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
        r = a
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")



def Hospital():
    city = Location.get()
    if city == "Bangalore":
        if p == q:
            t4.delete("1.0",END)
            t4.insert(END,Bangalore[p])

        elif q == r:
            t4.delete("1.0",END)
            t4.insert(END,Bangalore[q])

        elif p == r:
            t4.delete("1.0",END)
            t4.insert(END,Bangalore[r])

        else:
            t4.delete("1.0",END)
            t4.insert(END,"Incorrect")
            
    elif city == "Chennai":
        if p == q:
            t4.delete("1.0",END)
            t4.insert(END,Chennai[p])

        elif q == r:
            t4.delete("1.0",END)
            t4.insert(END,Chennai[q])

        elif p == r:
            t4.delete("1.0",END)
            t4.insert(END,Chennai[r])

        else:
            t4.delete("1.0",END)
            t4.insert(END,"Incorrect")
            
    elif city == "Hyderabad":
        if p == q:
            t4.delete("1.0",END)
            t4.insert(END,Hyderabad[p])

        elif q == r:
            t4.delete("1.0",END)
            t4.insert(END,Hyderabad[q])

        elif p == r:
            t4.delete("1.0",END)
            t4.insert(END,Hyderabad[r])

        else:
            t4.delete("1.0",END)
            t4.insert(END,"Incorrect")
    
    elif city == "Tirupati":
        if p == q:
            t4.delete("1.0",END)
            t4.insert(END,Tirupati[p])

        elif q == r:
            t4.delete("1.0",END)
            t4.insert(END,Tirupati[q])

        elif p == r:
            t4.delete("1.0",END)
            t4.insert(END,Tirupati[r])

        else:
            t4.delete("1.0",END)
            t4.insert(END,"Incorrect")
            
    elif city == "Vizag":
        if p == q:
            t4.delete("1.0",END)
            t4.insert(END,Vizag[p])

        elif q == r:
            t4.delete("1.0",END)
            t4.insert(END,Vizag[q])

        elif p == r:
            t4.delete("1.0",END)
            t4.insert(END,Vizag[r])

        else:
            t4.delete("1.0",END)
            t4.insert(END,"Incorrect")
        


@app.route('/')
def hello_world():
   return 'Hello World'

        
        

# In[15]:

print("root starting")
root = Tk()
root.configure(background='black')

Symptom1 = StringVar()
Symptom1.set("Select Here")

Symptom2 = StringVar()
Symptom2.set("Select Here")

Symptom3 = StringVar()
Symptom3.set("Select Here")

Symptom4 = StringVar()
Symptom4.set("Select Here sym")


Symptom5 = StringVar()
Symptom5.set("Select Here")

Location = StringVar()
Location.set("Select Here")

Name = StringVar()

w2 = Label(root, text="Multiple Disease Predictor using Machine Learning", fg="Red", bg="White")
w2.config(font=("Times",30,"bold italic"))
w2.grid(row=1, column=0, columnspan=2, padx=100)

w2 = Label(root, text="A Project by 22D12", fg="Pink", bg="Blue")
w2.config(font=("Times",30,"bold italic"))
w2.grid(row=2,column=0,columnspan=2, padx=100)

NameLb = Label(root, text="Name of the Patient", fg="Red", bg="Sky Blue")
NameLb.config(font=("Times",15,"bold italic"))
NameLb.grid(row=6, column=1, pady=15, sticky=W)

S1Lb = Label(root, text="Symptom 1", fg="Blue", bg="Pink")
S1Lb.config(font=("Times",15,"bold italic"))
S1Lb.grid(row=7, column=1, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="White", bg="Purple")
S2Lb.config(font=("Times",15,"bold italic"))
S2Lb.grid(row=8, column=1, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="Green",bg="white")
S3Lb.config(font=("Times",15,"bold italic"))
S3Lb.grid(row=9, column=1, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="blue", bg="Yellow")
S4Lb.config(font=("Times",15,"bold italic"))
S4Lb.grid(row=10, column=1, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="purple", bg="light green")
S5Lb.config(font=("Times",15,"bold italic"))
S5Lb.grid(row=11, column=1, pady=10, sticky=W)

wish = Label(root,text = "Take Care,Stay Safe and Keep Smiling...!",fg = 'white',bg = 'black')
wish.config(font=('Times',15,'bold italic'))
wish.grid(row =30,column= 0,columnspan=2, padx=100 )


OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

S1 = OptionMenu(root, Symptom1,*OPTIONS)
S1.grid(row=7, column=1)

S2 = OptionMenu(root, Symptom2,*OPTIONS)
S2.grid(row=8, column=1)

S3 = OptionMenu(root, Symptom3,*OPTIONS)
S3.grid(row=9, column=1)

S4 = OptionMenu(root, Symptom4,*OPTIONS)
S4.grid(row=10, column=1)

S5 = OptionMenu(root, Symptom5,*OPTIONS)
S5.grid(row=11, column=1)

LOCATIONS = sorted(cities)

C1 = OptionMenu(root,Location,*LOCATIONS)
C1.grid(row = 22, column = 0)

dst = Button(root, text="Prediction 1", command=DecisionTree,bg="Light green",fg="red")
dst.config(font=("Times",15,"bold italic"))
dst.grid(row=13, column=3,padx=10)

rnf = Button(root, text="Prediction 2", command=randomforest,bg="White",fg="purple")
rnf.config(font=("Times",15,"bold italic"))
rnf.grid(row=16, column=3,padx=10)

lr = Button(root, text="Prediction 3", command=NaiveBayes,bg="red",fg="white")
lr.config(font=("Times",15,"bold italic"))
lr.grid(row=19, column=3,padx=10)

hs = Button(root,text = 'Hospital',command = Hospital,bg = "Pink",fg = 'Blue')
hs.config(font=("Times",15,"bold italic"))
hs.grid(row=22,column = 3,padx = 10)

t1 = Text(root, height=1, width=40,bg="Light green",fg="red")
t1.config(font=("Times",15,"bold italic"))
t1.grid(row=13, column=1, padx=10)

t2 = Text(root, height=1, width=40,bg="White",fg="purple")
t2.config(font=("Times",15,"bold italic"))
t2.grid(row=16, column=1 , padx=10)

t3 = Text(root, height=1, width=40,bg="red",fg="white")
t3.config(font=("Times",15,"bold italic"))
t3.grid(row=19, column=1 , padx=10)

t4 = Text(root,height = 1,width = 100,bg = "pink",fg = "blue")
t4.config(font=("Times",15,"bold italic"))
t4.grid(row=22,column =1 ,padx = 15)

root.mainloop()





