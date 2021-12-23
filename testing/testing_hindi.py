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
messages=["Banti hai empowered woman, feminism pe gyan pelti hai aur din bhar roti rehti hai. Pahle rona band kar madarchod",
          "Agar koi bole ki ja ke chill maar to madarchod ki gand maar lene ka",
          "To Bhosdike tere Baap ka kya ja raha hai? tu Apna Ghar dekh na Madarchod",
          "SUNNY LEONE BANA KE CHODEGE TERI MAA KO SAMJH ME AAYA MADARCHOD NEEM KA PATTA KADVA HAI",
          "Screw the law of the land. If I find this chutiya Madarchod Mulla I will Lynch him, murder him, cut into millions of pieces",
          "Madarchod congresi or Bjp dono ki sarkar bani baad me wade bhul gye",
          " Teri maa Behan ko har rooj chood rahe hai tum khayal rakho .warnna aise hi chudawati rahati hai madarchod.",
          "Teri maa ko our Ben ko chudva sale benchod .... madarchod....Gabbar",
          "Tum madarchod Congressio ke raj mai fighter deal kyu nahi hue",
          " I never donate to any madarchod hindu priests, they are chors.",
          "Ab usko chhod mjse bat kr tera baap aa gya hai ab to ldki ko beech me q la ra hai Madarchod",
          "Teri Ammi k sath first inning Virat khela tha, tb tu paida hua, aur tera abbu Abhi us result ka gum mana rha h madarchod",
          "Madarchod jalta hai mujhse",
          "Kutte jite nahi madarchod dusro ko bhi nahi Jeene deti",
          "Teri maa ka bhosra ..bharve ki aulad ...madarchod...aukat me rahe ..nahi gand phar dunga ..bharve",
          "Madarchod shilpa randi rona band kar",
          "aur bhadwe ki aulad ho tum sale madarchod ki wajah se ye zait failta hai hindu muslim ka bhen ke",
          "Terrorist Madarchod humse baat karta hai",
          "abe Behsaram tu abhi tak mara nahi, tu to suiside karne waala tha kutte harami madarchod",
          "narendramodi arrey madarchod ky bacchey terko inko congrates karney ka time hain but ek tweet kar",
          " I m agree with you.. Behanchod madarchod log h bhosdi wali log.. Gali se hi baat smjjhane kaa",
          "Dosti ke namm par Sala dhoka deta hai",
          "Ye bahenchod, achhe khase Lund ko bekar karenge... itna bada risk le rahe hain",
          "Ladki k papa yaha par logo ko bahenchod bolkar gender equality me apna yogdan de",
          "ladkiyon ka kaam hota hi hai bus lund choosna",
          "mardo ko ladkiyon ki gaand aur boobe pasand hote hai",
          "darling kya tum mujhe apna doodh pilaogi",
          "Abe Bahenchod kuch To Sharam Kar Shakal Bhi Dekha Hai ",
          "Why getting lots of ad in your Wynk music app. Pure mood ki maabehen kr dete h",
          "Bahenchod. Aaukat mein rah", 
          "Bhai Me Apne Bete Ka Kya Naam Rakhu Jo Sabki Zubaan Pe Rahe",
          "Bhai Me Apne Bete Ka Kya Naam Rakhu Jo Sabki Zubaan Pe Rahe",
          "Are Bhai Apne Bahen Ka Nhi Banaya Kya Uska Bhi Bana Ke Dalna Tha Na Bahenchod Naale Me Paida Hua Kya",
          "Bahenchod this is the saddest thing, girl ready to do anything to help a guy sleep",
          "Bahenchod 65 saal TK apni maa chudwa rahe the tum log",
          "Bahenchod laal chaddi goons",
          "tum mujhe bilkul bhi achii nhi lagti ho",
          "oye lund ke beej aaj khane mein kya bna hai",
          "tu up bihar ka bhadva hoga jise gujarati me kadva kehete he ab to",
          "Pakistan bhadva hai",
          "Hindi nai aati to ma ch***vane aaya he politics me... sala bhadva .. samjate pappu tere perie ko",
          "Aur wo Bhadva hai Lalu Prasad \nSaala.ko thappad maar ke theek karo",
          "Jo ab bevakuf banane nikala hai murkh bhadva",
          " Kapil Sibbal ko Border pe Ulta Latka k itne CHHED kare ki Uski Randi-Maa b pehchan na sake",
          " sala bhadva chutiya Gujarati o ko gali deta he... Sorry Yar Mera Hindi thoda poor he",
          " kaye ko gaali de usko, chodthe Allah ke upar. ye musalmano",
          "Khujli bhikhari tatte, jhantu ka reporter, liar aur Jhantu aadmi party news, RNDTV ko chhodkr kabhi",
          "you are the ugliest hole one can have. Chutye marja, tere bap ne gunha kiya tuje peda karke. Bhadva kahin ka",
          "tu panoti hai, marja kahi",
          "naale mein doob jaa, suar kahi kaa",
          "Mujhey aapkee bahut yaad aaee",
          "Shubh raatri.",
          "aapka din achaa jaaye",
          "Kya mein aapki madad kar sakta hoon",
          "Kya aap meri madad kar saktey hain?",
          "Mein theek hoon, shukriya!",
          "Kyaa aap se mein baat kar saktey hoon",
          "aap bahut sundar hai chaand bhi sharma jaaye itna sundar face dekh ke to",
          "Aapsey milkar khushi huee",
          "Mujhey Hindi bhaashaa acchee lagtee hai",
          "Mein turant laut key aaoongaa",
          "Mein kabhi Bharat jaana chahoongaa",
          "Kyaa aap issey dohraa saktey hain",
          "Mujhey iss cheez kaa kuchh pata nahi hai",
          "han wo end tak frnd rhegi may be because kuch bhi ho wo ek dusre ka saath ni chordti",
          "kisi ki chahat mein itne pagal na hona ho sakta hy wo tumhari manzil na ho us ki muskurahat ko izhare mohabbat",
          "I love chai so much",
          "Kitney aadmi the",
          "Bade bade shehron mein aisi chhoti chhoti baatein hoti rehti hain, Senorita.",
          "Dosti ka ek usool hai, madam: no sorry, no thank you",
          "Kabhi Kabhi Kuch Jeetne Ke Liya Kuch Haar Na Padta Hai. Aur Haar Ke Jeetne Wale Ko Baazigar Kehte Hain.",
          "babuji aapka aashirwaad chahiye aaj mera exam hai",
          "tu mera sabse achaa dost hai",
          "upar ke room se bottle le aao",
          "mein class mein first aaya. mene bahut mehnnat ki hai",
          "mein wo hoon jo aapke saath tha",
          "wo khidki khulne kaa intzaar kar rha hai",
          "hum har roz nayi cheeze seekhti hain",
          "mene yeh file mein bahut mehnnat kari hai",
          "mein iss ladki se bahut pyaar karta hoon, mein isse shaadi karna chaahta hoon",
          "hum kaafi ache dost hai, hum ek saath hi khana khate hai",
          "humne aaj bahar dinner kiya tha",
          "books se kaafi knowledge milti hai, mein kaafi impressed hua",
          "kya aap mere liye door open kar sakte hai",
          "kya aap chai peena chahenge",
          "kya aap mereko bank tak kaa way bta sakte hai",
          "kya aap mere saath group bnana chahenge iss project ke liye",
          "aap khane mein kya khana chahte ho aur kab khana chahte ho",
          "old people ka aashirwaad hmare saath humsha rehta hain",
          "teachers hmare doosre bhagwaan hota hain",
          "tum mere kamre mein so sakte ho",
          "tum bahut cute ho, mann karta hai tumhe dil mein bassa loon",
          "aaj khane mein kya bna hai",
          "aaj mein aur sona chahta hoon",
          "kya aap mujhe ek achii book bta sakte hain",
          "knowledge hi ek aissi cheez hai jo marne ke baad bhi saath rehti hai",
          "tumse hi meri zindagi shuru hoti hai aur tum pe hi khatam",
          "bharat bahut sundar aur achaa desh hai",
          "humme har ek aurat kaa sammaan karna chahiye",
          "zindagi ek bahut khoobsurat gift hai humm humans ke liye"]
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




