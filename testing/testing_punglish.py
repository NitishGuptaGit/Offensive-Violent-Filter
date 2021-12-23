import pandas as pd
import nltk
import re
import sys
import numpy as np
sys.path.insert(0,r'C:\Users\Acer\Desktop\capstone_proj')
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import preprocess
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from cleaner import cleanall
df = pd.read_csv(r'C:\Users\Acer\Desktop\capstone_proj\libs\train.csv')
X=df.iloc[:,1].values
y=df.iloc[:,0].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=0)
training=preprocess(X_train)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(random_state=0,n_estimators=650)
clf.fit(training,y_train)
from sklearn.pipeline import Pipeline
text_clf=Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',RandomForestClassifier()),])
text_clf=text_clf.fit(X_train,y_train)
ans=text_clf.predict(X_test)
from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test,ans))
count1=0
messages=["Sat sri akaal",
         "Main theek haan",
         "Tu-adey nal mil kar bahut khusi hoi",
         "Tusi kithe dey ho?",
         "Thodi edi kimet ghatao ",
         "Aa tey boot mehnga hega hai ",
         "Meri smajhich nahi aanda ",
         "Jara rasta dasna",
         "Kripa karake iss nu phir kaho",
         "Gusalakhana kithe hai?",
         "Ki tusi angrezi bolde ho?",
         "Tuhanum mil ke khushi hoi",
         "Tuhaada Savagat hai",
         "Tussi meri madad karoge?",
         "Tussi kithey jaa rahe ho",
         "Main Wapis Jana Chahni aan",
         "Oyye Balle oyye Balle Dalbeer! Kitna bada hogaya rey tu!",
         "Chal Rehn Dey ",
         "Muh Dho Ke Aa",
         "Kiddan??",
         "Tuhaadaa kee haal hai jee?",
         "Dhannvaad meraa haal theek hai",
         "Mai tainu pyar karda haan ",
         "Baal Bachay theek aa?",
         "tusi bura na manno",
         "Fer milange. ",
         "Tusi menu maaf kro",
         "Mai roti khaani h",
         "Mere kol paise nahi h",
         "Tusi itho chale jaao",
         "Mai tuhade naal gal ni krni",
         "Menu kutteyan naal pyaar h",
         "Mai ikk doctor haan",
         "Tu peele rang di kameej paa",
         "Asi saare chalaange ghuman",
         "Mainu bahut kamm h",
         "Jaldi theek hojo hun",
         "Menu ajj bukhaar h",
         "Ithe sab tarah diyan dukaana ne",
         "Tusi ajj aae nahi",
         "Meri kitaab kithe h?",
         "Menu saag bahut pasand h",
         "Mai hun sona h",
         "Tu taan keh rahi si ki tu kamm krleya",
         "Saaf safai krlo",
         "Khaana bahut swaad h",
         "Menu dar lag reha hai",
         "Ajj mera janam din hai",
         "Tusi kithe si?",
         "Ohh kudi badi sohni h",
         "Dhaun Napp Deyange Tuhaddi",
         "Onda Phuddu Lagge Ai",
         "Shaitaan Diyaan Poonch",
         "Tuare Booth Pan Chate Hai",
         "Haye Rabba! Eho jehi aulad ton taan main be-auladi changi.",
         " Akhaan hegiyaan yaa button",
         " Kive mare hoye kutte vaang peya h",
         " Kehre khasam de naal gyi si",
         "Oo phuddi deya",
         " Teri bund paad deyaage",
         " Aaja kutteya ithe",
         "Changi bund bandook hoi ae",
         "Gaand ch khurak hundi h?",
         "Bhen chod tu kithe gyi si?",
         " Lunn baabe daa",
         " Hun lun phaad ke beh rao",
         " Teri maa nu keede pen",
         "Aa bada gaandu aa",
         "Ullu daa patha aa ki kita tu?",
         "Oo pehn di lunn sharam krjaa saaleya",
         "Tenu bund taar aai ae?",
         "Tu aap hi ungal laai honi",
         "Dhaun napp deyaange tuhadi",
         "Teri bhen di ",
         "Kithe muh maarda firdaa ae",
         "Khote de putt ethe aa",
         "Pendu jehi aurat",
         "Kaala baandar jeha mere kol naa aai",
         "Tu bhen chod mere naal galat kitaa",
         "Magron laath",
         "Gande naale ch jaa marr",
         "Tera beda gharak hove kutte da",
         "Khasma nu khao maa chudao",
         "Jithe di khoti uthe aa khaloti",
         "Peepni de muh aali",
         "Chauda naa ho zyada",
         "Kanjaraa kithe khalota firda ae",
         "Boo da boja",
         "Bootha pann deyage",
         "Banda marr jaave pr ginni naa",
         "Teri maa di khoteya sharam ni aaundi tenu?",
         "Bhen chod mere agge bolega?",
         "Baandar boothi kithe chaleya?",
         " Chittar pae jaane tere",
         "Chup krke andar jaa saaleya",
         "Bhen da takka chup karjaa",
         "Muh tod deyaangi tera",
         "Oye Fitteh Muh Tere Kangale!",
         "Abe hatt ehna lamba chauda khota hogea ae",
         "Gaand fatt ke hathaan ch aagi"]
for sample in messages:
    count1=count1+1
    print(str(count1)+": "+sample+".")
    review = re.sub('[^a-zA-Z]', ' ', sample)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #features=preprocess(cleaning)
    sample=review
    sample =np.array(sample)
    sizes=sample.shape[0]
    #print(sizes)
    print(sample)
    #sample=np.concatenate(sample)
    text=text_clf.predict(sample)
    print(text)
    count=0
    for i in range(0,sizes):
        if text[i]==1:
            count+=1
    prob=count/sizes
    if prob < 0.2:
        print("Message may not be hatespeech")
    else:
        print("Message is hatespeech")




