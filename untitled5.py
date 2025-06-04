# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:14:03 2024

@author: hugop
"""
echec_dim=8

class individu =:
    def_init_(self,val=None):
        if val==Nonne:
            self.val=random.sample("a remplir")
        else:
            self.b=val=val
            
            
def conflict(p1,p2):
    """return true si la reine à la position p1 est en conflit avec la reine en position p2"""
    return jsp

def fitness(self):
    """evaluer l'individu c est connaitre le nombre de conflit"""
    self.nbconflict =0
    for i in ....:
        for j in ):
    if(individu.conflict ([i,...],[j,....])):
        self.nbconflict=self.nbconflict+1
    return self.nbconflict

def create_rand_pop(count):
    
def evaluate(pop):
    
def selection(pop,hcount,lcount):
    
def croisement(ind1,ind2):

def mutation(ind):
    
def algoopSimple():
    pop=create_rand_pop(25)
    slutiontrouvee=False
    nbriteration=0
    while not solutiontrouvee:
        print("iteration numéro : ",nbriteration)
        nbriteration+=1
        evaluation=evaluate(pop)
        if evaluation[0].fitness()==0:
            solutiontrouvee=True
        else:
            select=selection (evalutation,10,4)
            croises=[]
            for i in range (0,len(select),2):
                croises+=croisement(select[i],select[i+1])
            mutes=[]
            for i in select : 
                mutes.append(mutation(i))
            newalea=create_rand_pop(5)
            pop=select[:]+croises[:]+mutes[:]+newalea[:]
    print(evaluation[0])
                