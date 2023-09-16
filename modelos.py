import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Input, Concatenate, BatchNormalization, Activation, GlobalAveragePooling1D, Flatten, MaxPool1D, LSTM, RepeatVector, TimeDistributed, SpatialDropout1D, Dropout
from Gates.Gates import GMU, mGMU

#<summary>
#   crearModelo: This function make a keras model with a feature extraction, feature union, and classification stages.
#                The layers to be use in each stage are define by the user. 
#   <param name="num_clases">
#       - The number of classes in the dataset.
#   </param>
#   <param name="window_size">
#       - Number of samples in each channel.
#   </param>
#   <param name="ejes">
#       - Number of Channels.
#   </param>
#   <param name="filtros">
#       - The number of filters used in each convolutional layer (inception layer included).
#   </param>
#   <param name="cant_entradas">
#       - Number of inputs (modalities) in the dataset-
#   </param>
#   <param name="extraccion">
#       - Feature extraction layer type to be used.
#         Options:
#           -  1I    : 1 inception layer.
#           -  1Conv : 1 convolutional layer.
#           -  21    : 2 inception layers.
#           -  2Conv : 2 convolutional layer.
#           -  2Ires : 2 inception layers with residual connection.
#   </param>
#   <param name="tipo_union">
#       - Feature fusion layer type to be used.
#         Options:
#           -  mGMU   : Minimal gated multimodal unit layer.
#           -  GMU    : Gated multimodal unit layer.
#           -  LSTM   : Long short-term memory layer.
#           -  Concat : Concatenation layer.
#   </param>
#   <param name="kernel_size_">
#       - Size of the kernel(filter) in convolutional layers.
#   </param>
#</summary>
def crearModelo(num_clases=7, window_size=100, ejes=3, filtros=32, cant_entradas=3, extraccion="1I", tipo_union="mGMU", kernel_size_ = 40):    
    extraccion_capa = []
    extraccion_output = []    
    extraccion_input = []
    [extraccion_capa.append(capas(window_size, ejes, filtros, "salida" + str(i+1), "entrada" + str(i+1), extraccion, kernel_size_ = kernel_size_)) for i in range(cant_entradas)]
    [extraccion_output.append(extraccion_capa[i].output) for i in range(cant_entradas)]
    [extraccion_input.append(extraccion_capa[i].input) for i in range(cant_entradas)]            
    if tipo_union == "mGMU" :
        union, z =  mGMU(cant_entradas, name="union")(extraccion_output)
        union_output = GlobalAveragePooling1D()(union)        
    elif tipo_union == "GMU" :
        union, z = GMU(cant_entradas, name="union")(extraccion_output)
        union_output = GlobalAveragePooling1D()(union)     
    elif tipo_union == "Concat" :
        union = Concatenate(axis = -1)(extraccion_output)
        union_output = GlobalAveragePooling1D()(union)
    elif tipo_union == "LSTM" :
        concat = Concatenate(axis = -1)(extraccion_output)
        union_output = LSTM(128)(concat)
    
    output  = Dense(num_clases, activation='softmax', name="salida_red")(union_output)    
    return Model([extraccion_input], output)

#<summary>
#   capas: Feature extraction stage maker.
#   <param name="window_size">
#       - Number of samples in each channel.
#   </param>
#   <param name="ejes">
#       - Number of Channels.
#   </param>
#   <param name="filtros">
#       - The number of filters used in each convolutional layer (inception layer included).
#   </param>
#   <param name="n_salida">
#       - Input(modality) number label for the output layer of the feature extraction layer. 
#   </param>
#   <param name="n_entrada">
#       - Input(modality) number label for the input layer of the feature extraction layer. 
#   </param>
#   <param name="tipo_capa">
#       - Feature extraction layer type to be used.
#         Options: (1I, 1Conv, 21, 2Conv, 2Ires)
#   </param>
#   <param name="kernel_size_">
#       - Size of the kernel(filter) in convolutional layers.
#   </param>
#</summary>
def capas(window_size, ejes, filtros, n_salida, n_entrada, tipo_capa, kernel_size_):
    ########################################################################
    ######################     1 Inception                      ###########
    ####################################################################### 
    if tipo_capa == "1I":     
        input_ = Input (shape=(window_size, ejes), name = n_entrada)
        bottleneck = Conv1D(filters=filtros, kernel_size=1, padding='same', activation='linear', use_bias=False)(input_)    
        conv_ = []
        filters_size = [kernel_size_  // (2 ** i) for i in range(3)]
        for i in filters_size:
            conv_.append(Conv1D(filters=filtros, kernel_size=i, strides=1, padding='same', activation='linear', use_bias=False)(bottleneck))
        max_pool = MaxPool1D(pool_size=ejes, strides=1, padding='same')(input_)
        conv_2 = Conv1D(filters=filtros, kernel_size=1, padding='same', activation='linear', use_bias=False)(max_pool)
        conv_.append(conv_2)
        concat = Concatenate(axis=-1)(conv_)
        batch = BatchNormalization()(concat)
        e_output = Activation(activation='relu', name=n_salida)(batch)    

    ########################################################################
    ######################     2 Inception                      ###########
    #######################################################################    
    elif tipo_capa == "2I":    
        input_ = Input (shape=(window_size, ejes), name = n_entrada)
        bottleneck = Conv1D(filters=filtros, kernel_size=1, padding='same', activation='linear', use_bias=False)(input_)    
        conv_ = []
        _kernel_size = [kernel_size_  // (2 ** i) for i in range(3)]
        for i in _kernel_size:
            conv_.append(Conv1D(filters=filtros, kernel_size=i, strides=1, padding='same', activation='linear', use_bias=False)(bottleneck))
        max_pool = MaxPool1D(pool_size=ejes, strides=1, padding='same')(input_)
        conv_2 = Conv1D(filters=filtros, kernel_size=1, padding='same', activation='linear', use_bias=False)(max_pool)
        conv_.append(conv_2)
        concat = Concatenate(axis=-1)(conv_)
        batch = BatchNormalization()(concat)
        activation = Activation(activation='relu')(batch)    
        sbottleneck = Conv1D(filters=filtros, kernel_size=1, padding='same', activation='linear', use_bias=False)(activation)    
        sconv_ = []    
        for i in _kernel_size:
            sconv_.append(Conv1D(filters=filtros, kernel_size=i, strides=1, padding='same', activation='linear', use_bias=False)(sbottleneck))
        smax_pool = MaxPool1D(pool_size=ejes, strides=1, padding='same')(activation)
        sconv_2 = Conv1D(filters=filtros, kernel_size=1, padding='same', activation='linear', use_bias=False)(smax_pool)
        sconv_.append(sconv_2)
        sconcat = Concatenate(axis=-1)(sconv_)
        sbatch = BatchNormalization()(sconcat)
        e_output = Activation(activation='relu', name=n_salida)(sbatch)                
         
    ########################################################################
    ######################     1 Convolutional                   ###########
    #######################################################################    
    elif tipo_capa == "1Conv": 
        input_ = Input (shape=(window_size, ejes), name = n_entrada)        
        e_output = Conv1D(name=n_salida, filters=filtros, kernel_size=(kernel_size_ // 2) , strides=1, padding='same', activation='linear', use_bias=False)(input_)                
        
    ########################################################################
    ######################     2 Convolutional                   ###########
    #######################################################################   
    elif tipo_capa == "2Conv":
        input_ = Input (shape=(window_size , ejes), name = n_entrada)        
        conv_ = Conv1D(filters=filtros, kernel_size=(kernel_size_ //2), strides=1, padding='same', activation='linear', use_bias=False)(input_)    
        e_output = Conv1D(name=n_salida, filters=filtros, kernel_size=kernel_size_ , strides=1, padding='same', activation='linear', use_bias=False)(conv_)
        
    ########################################################################
    ######################     2 inception con residual          ###########
    #######################################################################   
    elif tipo_capa == "2Ires":    
        input_ = Input (shape=(window_size, ejes), name = n_entrada)
        bottleneck = Conv1D(filters=filtros, kernel_size=1, padding='same', activation='linear', use_bias=False)(input_)    
        conv_ = []
        _kernel_size = [kernel_size_ // (2 ** i) for i in range(3)]
        for i in _kernel_size:
            conv_.append(Conv1D(filters=filtros, kernel_size=i, strides=1, padding='same', activation='linear', use_bias=False)(bottleneck))
        max_pool = MaxPool1D(pool_size=ejes, strides=1, padding='same')(input_)
        conv_2 = Conv1D(filters=filtros, kernel_size=1, padding='same', activation='linear', use_bias=False)(max_pool)
        conv_.append(conv_2)
        concat = Concatenate(axis=-1)(conv_)
        batch = BatchNormalization()(concat)
        activation = Activation(activation='relu')(batch)    
        sbottleneck = Conv1D(filters=filtros, kernel_size=1, padding='same', activation='linear', use_bias=False)(activation)    
        sconv_ = []    
        for i in _kernel_size:
            sconv_.append(Conv1D(filters=filtros, kernel_size=i, strides=1, padding='same', activation='linear', use_bias=False)(sbottleneck))
        smax_pool = MaxPool1D(pool_size=ejes, strides=1, padding='same')(activation)
        sconv_2 = Conv1D(filters=filtros, kernel_size=1, padding='same', activation='linear', use_bias=False)(smax_pool)
        sconv_.append(sconv_2)
        sconcat = Concatenate(axis=-1)(sconv_)
        sbatch = BatchNormalization()(sconcat)
        sactivation = Activation(activation='relu')(sbatch)    
        e_output = Concatenate(name=n_salida, axis=-1)([activation,sactivation])
            
    return Model(input_, e_output)  




