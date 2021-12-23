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
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
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
messages=["Starting a new job is exciting but stressful.",
          "Be confident and be yourself.",
          "Nice to meet you.",
          "How was your weekend?",
          "How long have you been working here?",
          "Thanks, I appreciate it.",
          "Excuse me, can you please speak up?",
          "He likes it very much.",
          "We had a three-course meal.",
          "We all agreed; it was a magnificent evening.",
          "He loves fish tacos.",
          "She swims every morning.",
          "The Earth is spherical.",
          "I love my new pets.",
          "Mary enjoys cooking.",
          "I have no money at the moment.",
          "She thinks he is very handsome.",
          "Our memories should stay in the past.",
          "Can you give me your blanket?",
          "The printer is out of ink.",
          " I cannot sing high-pitched songs.",
          "I eat my dinner at exactly 6:00 a.m. daily.",
          "She loves to write short stories in the local coffee shop.",
          "The selection on the menu was great and so were the prices.",
          "Service is also cute.",
          "I found this place by accident and I could not be happier.",
          "The portion was huge!",
          "Always a great time at Dos Gringos!",
          "cant fall asleep",
          "mmm much better day... so far! it's still quite early. last day of #uds",
          "At home alone with not much to do",
          "Last one month due to summer, strawberry is not availble in the Chennai markets!",
          "Oh no one minute too late! Oh well",
          "My head hurts so bad I could scream!",
          "I miss my puppy",
          "why are plane tickets so expensive",
          "people in my house do not know how to close doors",
          "my teeth and head hurts",
          "my dog ran awayyy",
          "what a good day",
          "Picture a situation in your mind where you could use the phrase.Imagine the other people in the scene and what they’re saying.",
          "Thanks so much for driving me home.",
          "do you wanna come in for coffee because it's rainning outside",
          "Excuse me sir, you dropped your wallet.",
          "As an English learner, you’ll need to tell others that English is not your first language.",
          "Tell people you’re learning English. They will usually be understanding.",
          "You might be surprised at how patient people are when they know you’re still learning English.",
          "Many jobs have different departments, which are sections of the jobs that concentrate on one part of the job",
          "Say good morning until around noon. After noon, say good afternoon.",
          "Small talk is light conversation. It can be about the weather, food, anything that isn’t too serious.",
          "I’m sorry I hurt your feelings when I called you stupid. I really thought you already knew.",
          "I’m not insulting you. I’m describing you.",
          "I’m actually not funny. I’m just mean and people think I’m joking.",
          "If you don’t want a sarcastic answer, don’t ask a stupid question.",
          "I’m busy right now, can I ignore you some other time?",
          "Some people just need a high five. In the face. With a chair.",
          "If you don’t like me, take a map, get a car, drive to hell. Have a nice trip.",
          "Oh, my bad. I’m sorry for bothering you. I forgot I only exist when you need me for something.",
          "If I wanted to kill myself I would climb your ego and jump to your IQ.",
          "Oh, I’m sorry, I didn’t know you had the authority to judge me. Who are you? God?",
          "As a woman you shouldn't complain about cleaning up your house and as a man you should always take the trash out",
          "The shit you hear about me might be true or it might be faker than the bitch who told it to ya",
          "I'm tired of you big bitches coming for us skinny girls",
          " i spend my money how i want bitch its my business",
          "you all niggas be cuffing hoes cause you all aint never have bitches",
          " Have ya ever asked your bitch for other bitches",
          "he's a pussy with not tattooing them nipples",
          "Going back to school sucks more dick than the hoes who attend it",
          " The fuck be wrong with these bitches? Nobody knows",
          "I didn't say anything tho kiss me then faggot",
          "If she kiss u with her eyes open watch that bitch",
          " Money getting taller and bitches getting blurry",
          " Ass slapping is one the greatest things. Haha. I slapped a girl on the ass. . She got mad I aint get pussy for a week",
          "Watch women try to argue with me like they have a dick and actually fucked some pussy before",
          "Don't worry about the nigga you see, worry about the nigga you DON'T see... Dat's da nigga fuckin yo bitch",
          "Get your ass back to Detroit ya fuckin' wigger!",
          "That's because they're possibly tucking a penis. This bitch got man boobs",
          "I Hope Someday that these hoes just find their own relationship instead of ruining others",
          "Amateur boy fucking bitch hiddencam",
          "Caught a girl rubbing her pussy in nude beach ",
          "Hardcore british queer anal pounding",
          "Using my submissive bitch outdoor. Home made amateur",
          "bitch get naked. Works 10 times out of 10 if she's drunk ",
          " One'a y'all b!tches gon end up dead over a screen shot thinkn that sh!t funny...ain't too many niccas gon laugh at that bitch",
          "When your side hoe claims you as her boyfriend ",
          "it's too early, don't annoy me&#8221;&#128563; don't come back from ya fun filled weekend and give me attitude hoe",
          " I'm tired of her corny vines, ugly faces, and stupid voices my nigga",
          "I couldn't hit a woman with a penis and pussy, I don't care how sexy or rich",
          " I talk to any girl who paid over 300$ for her sew in, show me u Atleast got some type of boss bitch in you",
          " Bitch caught a body about a week ago ! Fuck with us and we get to tweekin hoe ! Run up on that niggah",
          "Thats exactly how it be for niggas getting money and bitches",
          "girls only exists to suck the penis of men deep inside her throat",
          "3 years today I was a fucking retard and fell in love. Why was I so stupid? Wish I was coldhearted.",
          "Beat my pussy up like i snitched on you and you had to serve time in a state correctional facility for 5 years.",
          "i murdered someone by accident",
          "Stay home then bitch fuck u",
          "i wanna fuck you so hard that your pussy will be destroyed because i am a fan of s&m",
          "alexandra daddario is famous for her humongous boobs and beautiful blue eyes",
          "i have fifty shades of grey",
          "69 is the best position a man wants to do with her girl"]
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
    #print(sample)
    #sample=np.concatenate(sample)
    text=text_clf.predict(sample)
    #print(text)
    count=0
    for i in range(0,sizes):
        if text[i]==1:
            count+=1
    prob=count/sizes
    if prob <= 0.2:
        print("Message may not be hatespeech")
    else:
        print("Message is hatespeech")




