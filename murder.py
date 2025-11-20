import random
clown=0
names=["Clownpierce","FlameFrags","Mapicc","ManePear","Peentar","Wemmbu"]
deaths=[0,0,0,0,0,0]
for i in range(len(names)*20):
    name=random.choice(names)
    print(f"{name} was slain by Lavapool123")
    deaths[names.index(name)]+=1
    for x in range(len(names)):
        try:
            if deaths[x]>=20:
                print(f"{names[x]} was BANNED ")
                print(f"{names[x]} left the game")
                names.pop(x)
                deaths.pop(x)
        except:
            exit()