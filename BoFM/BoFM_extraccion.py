from stringprep import in_table_a1
import numpy as np
import math
from random import randint
from sklearn.ensemble import RandomForestClassifier
import gc


#####################################################################################
#################                 BoFM Histograms       #############################
#####################################################################################
    #<summary>
    #   This class implements the Bag of Features algorithm for a multivariate time 
    #   series. This implementations is an extension of the algorithm proposed by 
    #   Baydogan et al. In this implementation every channel of every input train two  
    #   randomForest classifier (one for the CPE used to make the codebook)
    # 
    #   THE IMPLEMENTATION ONLY IMPLEMENTS THE FIRST STAGE OF THE ALGORITHM TO MAKE 
    #   THE HISTOGRAM
    #     
    #</summary>
class BoF:
    #<summary>
    #   Global variables
    #   <var name="setPuntos">
    #       - Flag that change when the position of the rxd subsequence it's chosen. 
    #   </param>
    #   <var name="setFit">
    #       - Flag that changes when the training process is complete.
    #   </param>
    #</summary>
    setPuntos=False
    setFit=False

    #<summary>
    #   Constructor
    #   <param name="d">
    #       - Number of intervals d for each subsequence  
    #   </param>
    #   <param name="z">
    #       - proportion(0<z<1) of the length of the time series used to set the lower bound on the subsequence length.
    #   </param>
    #   <param name="w_min">
    #       - Minimum interval length
    #   </param>
    #   <param name="b">
    #       - Number of bins to discretize the CPEs for the codebook.
    #   </param>
    #   <param name="n_clases">
    #       - Number of classes in the dataset. When its empty, it can be calculated automatically. 
    #   </param>
    #   <param name="ntree">
    #       - Number of trees to be used in the RandomForest classifier.
    #   </param>
    #   <param name="njobs">
    #       - Numbre of threads to be used in the RandomForest classifier.
    #   </param>
    #   <param name="clf">
    #       - Classifier to be used (Only RandomForest classifier can be used in this implementation).
    #   </param>
    #   <param name="mid">
    #       - min_impurity_decrease in the RandomForest classifier.
    #   </param>
    #   <param name="tipo">
    #       - Subsequence generating mode.
    #         Options:
    #           -  2: Subsequences are generated in a fixed, uniform manner.
    #           -  3: Subsequences are generated with random lengths in random positions.
    #   </param>
    #</summary>
    def __init__(self, d=None,z=.1, w_min=5, b=10,n_clases=None,ntree=50,njobs=2, clf="RandomForest", mid=0, tipo=3):    
        self.mid=mid
        self.clf=clf
        self.njobs=njobs    
        self.ntree=ntree
        self.n_clases=n_clases
        self.d=d   #Numero de intervalos para una subsecuencia            
        self.z=z            
        self.w_min=w_min
        self.b=b
        self.forests = []
        self.setPuntos==False
        self.tipo = tipo        

    #<summary>
    #   fit:  Starts the training process of the BoFM algorithm.
    #   <var name="X">
    #       - Multivariate Time series data source. (Training data)
    #   </param>
    #   <var name="Y">
    #       - Multivariate Time series data target. (Training data) <- Ground truth
    #   </param>
    #</summary> 
    def fit(self, X, Y):   
        self.channels =  X.shape[2] 
        X.astype(np.double)
        if self.d==None:                
                T=X[0].shape[0]  # If d is empty is calculated from the length of the first series.                                                       
                self.d=math.floor((self.z*T)/self.w_min)            
        for k in range(self.channels):
            self.forests.append(RandomForestClassifier(n_estimators=self.ntree, criterion='gini',n_jobs=self.njobs, warm_start=True, min_impurity_decrease=self.mid, oob_score=True))
        f_train_t = []
        for k in range(self.channels):
            f_train_t.append(self.vectorizacion(X[:,:,k], Y))    
        h_train_t = []
        for k in range(self.channels):
            histograma = self.aHistrogramas(f_train_t[k], X.shape[0], self.forests[k])
            histograma = np.expand_dims(histograma, axis =2)
            h_train_t.append(histograma)
        self.h_train_set= np.concatenate(h_train_t, axis=2)
        del h_train_t
        del f_train_t
        gc.collect()        
        self.setFit=True    

    #<summary>
    #   transform:  Transform test data to histograms (1st stage of the algorithm).
    #   <var name="X">
    #       - Multivariate Time series data source. (Test data)
    #   </param>
    #</summary> 
    def transform(self, X):
        if self.setFit:
            self.channels =  X.shape[2] 
            Y=np.zeros((X.shape[0],1),dtype=int)
            f_test_t = []
            for k in range(self.channels):
                f_test_t.append(self.vectorizacion(X[:,:,k], Y))                
            h_test_t = []            
            for k in range(self.channels):
                histograma = self.aHistrogramas(f_test_t[k], X.shape[0], self.forests[k], fase="test")
                histograma = np.expand_dims(histograma, axis = 2)
                h_test_t.append(histograma)            
            h_test = np.concatenate(h_test_t, axis=2)                     
            del h_test_t
            del f_test_t
            gc.collect()
            return h_test
        else:
            print("No se ha entrenado, utilizar funcion fit()")

    #<summary>
    #   getHtrain:  Returns the histograms of the training set (1st stage).
    #</summary>   
    def getHtrain(self):
        return self.h_train_set

    #<summary>
    #   transform:  Transform test data to histograms (1st stage of the algorithm).
    #   <var name="X">
    #       - Multivariate Time series data source. (Test data)
    #   </param>
    #</summary> 
    def aHistrogramas(self,Vector,n_series, Rf1,fase="train"):                
        indice_clase=(self.d*3)+4
        indice_id=(self.d*3)+5
        indice_caracteristicas=(self.d*3)+3        
        arreglo_clases=Vector[:,indice_clase:(indice_clase+1)]
        arreglo_clases=arreglo_clases.astype(int)
        arreglo_id=Vector[:,indice_id:(indice_id+1)]
        arreglo_id=arreglo_id.astype(int)
        arreglo_caracteristicas=Vector[:,0:(indice_caracteristicas+1)]        
        if fase == "train" and self.n_clases==None:
            self.n_clases=np.unique(arreglo_clases).shape[0]        
        arreglo_clases=arreglo_clases.ravel()                
        if fase == "train":            
            tolerance=0.05                       
            arreglo_caracteristicas=arreglo_caracteristicas.round(decimals=7)
            Rf1.fit(arreglo_caracteristicas, arreglo_clases)                    
            prev_OOBerror=1                                  
            """
            oob_score_ : float
            Score of the training dataset obtained using an out-of-bag estimate.
            oob_decision_function_ : array of shape = [n_samples, n_classes]
            Decision function computed with out-of-bag estimate on the training set. If n_estimators is small it might
            be possible that a data point was never left out during the bootstrap. In this case, oob_decision_function_ 
            might contain NaN.
            """
            cur_OOBerror=1-Rf1.oob_score_
            itera=0        
            while (itera<20) and (cur_OOBerror<((1-tolerance)*prev_OOBerror)):                
                prev_OOBerror=cur_OOBerror    
                Rf1.n_estimators +=self.ntree                
                Rf1.fit(arreglo_caracteristicas, arreglo_clases)                                
                cur_OOBerror=1-Rf1.oob_score_
                itera +=1                                 
        if fase == "train":    
            CPE= np.nan_to_num(Rf1.oob_decision_function_)
        else:
            CPE=Rf1.predict_proba(arreglo_caracteristicas)
        h_series=np.zeros((n_series,self.n_clases*(self.b+1)),dtype=int)    
        arreglo_id=arreglo_id.transpose()[0]
        get_indices = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]                                
        for i in range (n_series):            
            indices_serie=get_indices(i,arreglo_id)                                    
            for j in indices_serie:                            
                val=-1   #almacener el promedio mayor
                for k in range(self.n_clases):                                                                
                    conj_palabra=self.getPalabra(CPE[j][k],i,j,k) 
                    palabra=int(k*self.b+conj_palabra)                    
                    h_series[i][palabra] += 1    
                    if CPE[j][k]>val:
                        f_rel=k
                        val=CPE[j][k]
                h_series[i][self.n_clases*self.b + f_rel] += 1                
        return h_series            
    
    #<summary>
    #   getPalabra:  Internal use, function to discretize.
    #   <var name="valor">
    #       - Value to discretize.
    #   </param>
    #</summary> 
    def getPalabra(self,valor,i,j,k):    
        temp=math.floor(valor*self.b)
        if temp== self.b:            
            temp-=1            
        return temp

    #<summary>
    #   vectorizacion:  Internal use, vector generator.
    #   <var name="X">
    #       - Multivariate Time series data source. (Training data)
    #   </param>
    #   <var name="Y">
    #       - Multivariate Time series data target. (Training data) <- Ground truth
    #   </param>
    #</summary> 
    def vectorizacion(self,X,Y):                
        vectorizados=[]             
        for i in range (X.shape[0]):
            T=X[i].shape[0]
            l_min=math.floor(T*self.z)            
            n_sub=math.floor((T/self.w_min) -self.d)            
            if l_min < (self.d*self.w_min):
                print("longitud minima de la subsecuencia demasiado pequeña")                                
                break  
            if i==0 and self.setPuntos==False:
                self.puntos=self.generarPuntos(X[i].shape[0],l_min,n_sub)
                self.setPuntos=True		
            puntos=self.puntos		
            if self.tipo==3:	
                puntos=self.generarPuntos(X[i].shape[0],l_min,n_sub)  
            vectores_serie=self.f_extraccion(Y[i],X[i],puntos,i)            
            vectorizados.append(vectores_serie)
        vectorizados=np.concatenate( vectorizados, axis=0 )   
        return vectorizados

    #<summary>
    #   f_extraccion:  Internal use, Generates the statistics representation of the subsequences.
    #   <var name="clase">
    #       - Class of the subsequence.
    #   </param>
    #   <var name="serie">
    #       - Id of the original vector.
    #   </param>
    #   <var name="puntos">
    #       - Position of the subsequences.
    #   </param>
    #</summary> 
    def f_extraccion(self,clase,serie,puntos,id_serie):
        resp=np.zeros((puntos.shape[0],(self.d*3)+6), dtype=np.double)                    
        for x in range (puntos.shape[0]):            
            s=0            
            for y in range (self.d):
                punto_inicial=int(math.floor(y*(((puntos[x][1])-puntos[x][0])/self.d)+puntos[x][0]))
                punto_final=int(math.floor((y+1)*(((puntos[x][1])-puntos[x][0])/self.d)+puntos[x][0]))                                        
                resp[x][s]=self.Slope(serie[punto_inicial:punto_final],punto_inicial,punto_final)
                resp[x,s+1]=np.mean(serie[punto_inicial:punto_final],axis=0)
                resp[x][s+2]=np.var(serie[punto_inicial:punto_final],axis=0)                
                s=s+3                
            resp[x][s]=np.mean(serie[puntos[x][0]:puntos[x][1]+1],axis=0)
            resp[x][s+1]=np.var(serie[puntos[x][0]:puntos[x][1]+1],axis=0)                
            resp[x][s+2]=puntos[x][0]
            resp[x][s+3]=puntos[x][1]
            resp[x][s+4]=clase
            resp[x][s+5]=id_serie
        return resp

    #<summary>
    #   generarPuntos:  Internal use, Generates the position of the subsequences.
    #   <var name="c_puntos_serie">
    #       - Interval of the subsequence.
    #   </param>
    #   <var name="l_min">
    #       - lower bound on the subsequence length.
    #   </param>
    #   <var name="n_sub">
    #       - Number of subsequence to be made.
    #   </param>
    #</summary> 
    def generarPuntos(self,c_puntos_serie,l_min,n_sub):            
        resp=np.zeros((n_sub,2),dtype=int)    
        if self.tipo==2:     			
            #se generan ventanas seguidas una de otra			
            for i in range (n_sub):
                resp[i][0]=self.w_min*i
                resp[i][1]=self.w_min*i+l_min		
        else:
            for i in range (n_sub):
                resp[i][0]=randint(0,(c_puntos_serie-l_min))
                final=math.floor((c_puntos_serie-resp[i][0])/self.d)                
                final=randint(self.w_min,final)
                final=final*self.d
                resp[i][1]=resp[i][0]+final        
        return resp

    #<summary>
    #   Slope:  Internal use, Slope calculation. 
    #   <var name="serie">
    #       - Lenght of the time series.
    #   </param>
    #   <var name="punto_inicial">
    #       - Initial position of the interval.
    #   </param>
    #   <var name="punto_final">
    #       - Final position of the interval.
    #   </param>
    #</summary> 
    def Slope(self,serie,punto_inicial,punto_final):
        x=list(range(punto_inicial,punto_final))
        ymean = np.mean(serie,axis=0)
        x=np.asarray(x)            
        xmean=np.mean(x,axis=0)
        sx=0
        sxy=0
        for i in range(x.shape[0]):            
            sx=sx+math.pow((x[i]-xmean),2)            
            sxy=sxy+(x[i]-xmean)*(serie[i]-ymean)                
        if sx==0 and sxy==0:
            print ("pi:",punto_inicial," pf:",punto_final)        
        return sxy/sx