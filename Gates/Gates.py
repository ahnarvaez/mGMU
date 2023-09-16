import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

#####################################################################################
#################                 mGMU                 #############################
#####################################################################################
class mGMU(Layer):
    #<summary>
    #   Constructor
    #   <param name="units">
    #       - Number of inputs (modalities). In this approach all modalities need to 
    #         have same dimension.  
    #   </param>
    #</summary>
    def __init__(self, units, **kwargs):
        self.units = units        
        super(mGMU, self).__init__(**kwargs)

    #<summary>
    #   build:This method is called once by the Constructor. The weigs matrix, vars 
    #         and const are buid here.
    #       - The weight matrix of the z neurons is Wz. It has a dimension of Sxn.
    #         Where S its the number of samples, and n the number of inputs. 
    #         (we use a single matrix for al the z neurons). 
    #       - For every h we use a matrix of size CxC. Where C its the number os 
    #         channels used. All the matrix were encapsulated in a list called W. 
    #   <param name="input_shape">
    #       - Dimension of the input. For multiple inputs this must be a list.
    #         This parameter is automatically extracted from the output of 
    #         previous layer  
    #   </param>
    #</summary>
    def build(self, input_shape):        
        assert isinstance(input_shape, list)        
        self.eje = -1                
        self.Wz = self.add_weight(shape=(input_shape[0][self.eje]*self.units, self.units),
                                        initializer='uniform',
                                        name='Wz',
                                        trainable=True)
        self.dim  = input_shape[0][self.eje]
        self.dim2 = input_shape[0][-2]
        self.W =[]
        for i in range(self.units):        
            name="W_"+str(i)
            self.W.append(self.add_weight(shape=(input_shape[0][self.eje],input_shape[0][self.eje]),
                                    initializer='uniform',
                                    name=name,
                                    trainable=True))
        self.built = True   

    #<summary>
    #   call: The equations that governs the gate are here. 
    #   <param name="list">
    #       - List with all the inputs (modalities) 
    #   </param>
    #</summary>
    def call(self, inputs):
        assert isinstance(inputs, list)            
        x = K.concatenate(inputs, axis=self.eje)
        h = []                  
        # -  We need to broadcast all vectors in Wz matrix in order to get the 
        #    dimension of each input (SxC). In addition we need to broadcast 
        #    to the batch size.    
        #   hij = tanh(Wi⋅Xij), for i=1,...,n, and j=1,...,C        
        for i in range(self.units):
            broadcast_shape = tf.where([True, False, False], tf.shape(inputs[i]), [0, self.dim, self.dim])
            W_ = tf.broadcast_to(self.W[i], broadcast_shape) 
            h.append(tf.math.tanh(tf.matmul(inputs[i], W_)))
        #   zi = σ ([Xi,...,Xn]⋅[Wzi,..., Wzi(ntimes)]), for i = 1,...,n
        Wz_ = tf.broadcast_to(self.Wz, [tf.shape(x)[0], self.dim*self.units, self.units])                     
        z = tf.math.sigmoid(tf.matmul(x, Wz_))                
        repeat = []
        for i in range(self.units):
            repeat.append(z[:,:,i])            
            repeat[i] = tf.expand_dims(repeat[i], -1)    
            repeat[i] = tf.keras.backend.repeat_elements(repeat[i], self.dim , -1) 
        resp = repeat[0]*h[0]
        #   Ot = sum(zi*hi)
        for i in range(1,self.units):
            resp += repeat[i]*h[i]
        return resp, z # <- For z weights
                
    
    #<summary>
    #   compute_output_shape: Dimension of the output of the gate.
    #   <param name="input_shape">
    #       - Dimension of the input. For multiple inputs this must be a list.
    #         This parameter is automatically extracted from the output of 
    #         previous layer 
    #   </param>
    #</summary>
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)        
        return [input_shape[0][-2], input_shape[0][-1]]
    
    #<summary>
    #   get_config: From the guide; will return a Python dict containing 
    #               the configuration of the model or layer, respectively. 
    #               Need it to rebuild the model.
    #   * this function does not need params to be called, but the 
    #     parameters used by the constructor must be here.
    #</summary>
    def get_config(self):
        config = super(mGMU, self).get_config()
        config.update({
            'units': self.units            
        })
        return config 

#####################################################################################
#################                      GMU           ################################
#####################################################################################
class GMU(Layer):
    #<summary>
    #   Constructor
    #   <param name="units">
    #       - Number of inputs (modalities). In this approach all modalities need to 
    #         have same dimension.  
    #   </param>
    #</summary>
    def __init__(self, units, **kwargs):
        self.units = units        
        super(GMU, self).__init__(**kwargs)

    #<summary>
    #   build:This method is called once by the Constructor. The weigs matrix, vars 
    #         and const are buid here.
    #       - The weight matrix of the z neurons is Wz. It has a dimension of S(xn)xC.
    #         Where S its the number of samples,C the number of Channels , and n 
    #         the number of inputs. (we use a single matrix for al the z neurons). 
    #       - For every h we use a matrix of size CxC. Where C its the number os 
    #         channels used. All the matrix were encapsulated in a list called W. 
    #   <param name="input_shape">
    #       - Dimension of the input. For multiple inputs this must be a list.
    #         This parameter is automatically extracted from the output of 
    #         previous layer  
    #   </param>
    #</summary>
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.eje = -1
        self.Wz=self.add_weight(shape=(input_shape[0][self.eje]*self.units,input_shape[0][self.eje]*self.units),
                                    initializer='uniform',
                                    name='Wz',
                                    trainable=True)
        self.dim = input_shape[0][self.eje]
        self.W =[]
        for i in range(self.units):        
            name="W_"+str(i)
            self.W.append(self.add_weight(shape=(input_shape[0][self.eje],input_shape[0][self.eje]),
                                    initializer='uniform',
                                    name=name,
                                    trainable=True))        
        self.built = True        

    #<summary>
    #   call: The equations that governs the gate are here. 
    #   <param name="list">
    #       - List with all the inputs (modalities) 
    #   </param>
    #</summary>
    def call(self, inputs):
        assert isinstance(inputs, list)            
        x = K.concatenate(inputs, axis=self.eje)
        h = []     
        # -  We need to broadcast to the batch size.    
        #   hi = tanh(Wi⋅Xij), for i=1,...,n
        for i in range(self.units):
            broadcast_shape = tf.where([True, False, False], tf.shape(inputs[i]), [0, self.dim, self.dim])
            W_ = tf.broadcast_to(self.W[i], broadcast_shape)
            h.append(tf.math.tanh(tf.matmul(inputs[i], W_)))
        broadcast_shape = tf.where([True, False, False], tf.shape(x), [0, self.dim*self.units, self.dim*self.units])
        Wz = tf.broadcast_to(self.Wz, broadcast_shape)  
        # zi = σ ([Xi,...,Xn]⋅Wzi), for i = 1,...,n              
        z = tf.math.sigmoid(tf.matmul(x, Wz))
        # Ot = sum(zi*hi)
        resp = z[:,:,0:inputs[0].shape[self.eje]]*h[0] 
        for i in range(1,self.units):
            resp +=z[:,:,i*(inputs[0].shape[self.eje]):(inputs[0].shape[self.eje]*(i+1))]*h[i]
        return resp, z
                
    #<summary>
    #   compute_output_shape: Dimension of the output of the gate.
    #   <param name="input_shape">
    #       - Dimension of the input. For multiple inputs this must be a list.
    #         This parameter is automatically extracted from the output of 
    #         previous layer 
    #   </param>
    #</summary>
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)        
        return [input_shape[0][-2],input_shape[0][-1]]
    
    #<summary>
    #   get_config: From the guide; will return a Python dict containing 
    #               the configuration of the model or layer, respectively. 
    #               Need it to rebuild the model.
    #   * this function does not need params to be called, but the 
    #     parameters used by the constructor must be here.
    #</summary>
    def get_config(self):
        config = super(GMU, self).get_config()
        config.update({
            'units': self.units            
        })
        return config  

